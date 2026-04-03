"""
Proactive Cold Start Mitigation in Serverless Environments
==========================================================
Full 31-Day Pipeline — PyTorch Edition (Python 3.14 compatible)
Dataset: Huawei Public Cloud Trace 2025 – Region 1 Cold Start Traces
----------------------------------------------------------------

Architecture:
  1. Bidirectional LSTM (PyTorch)  — sequential temporal branch
  2. RandomForest (scikit-learn)   — tabular feature branch
  3. Confidence-weighted ensemble  — weighted by validation AUC
  4. Adaptive Threshold Controller — online FP/FN-based adaptation
  5. Multi-horizon prediction      — 1-min, 5-min, 15-min targets

Usage:
    First unzip the R1.zip file to get the CSV files (day_0.csv, day_1.csv, ..., day_30.csv).
    python main.py --dir /path/to/data/
"""

import os, sys, glob, pickle, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Fixed research-grade splits — explicitly defined for reproducibility
# Train: days 0–18  (19 days)
# Val  : days 19–24  (6 days)
# Test : days 25–30  (6 days, held-out evaluation)
TRAIN_DAYS = list(range(0, 19))    # days 0–18
VAL_DAYS   = list(range(19, 25))   # days 19–24
TEST_DAYS  = list(range(25, 31))   # days 25–30

COLD_THRESHOLD = 0.25
SEQUENCE_LEN   = 30
LSTM_EPOCHS    = 120      # increased: more epochs since we have 31 days of data
BATCH_SIZE     = 64
PATIENCE       = 20       # increased: allow more epochs without improvement before stopping

HORIZONS = {
    'target_1min':  1,
    'target_5min':  5,
    'target_15min': 15,
}
PRIMARY_HORIZON = 'target_5min'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TABULAR_FEATURES = [
    'hour', 'is_business', 'day_of_week',
    'roll_mean_5',  'roll_std_5',  'roll_cold_5',
    'roll_mean_15', 'roll_std_15', 'roll_cold_15',
    'roll_mean_60', 'roll_std_60', 'roll_cold_60',
    'lag_total_1',  'lag_cold_1',
    'lag_total_5',  'lag_cold_5',
    'lag_total_15', 'lag_cold_15',
    'lag_total_30', 'lag_cold_30',
    'trend_5', 'trend_15',
    'burst_ratio',
    'cold_acceleration',
]

SEQUENCE_FEATURES = [
    'total_norm', 'cold_rate',
    'roll_mean_5',  'roll_cold_5',
    'roll_mean_15', 'roll_cold_15',
    'trend_5',
]


# ═══════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_day(path: str) -> pd.DataFrame:
    """Load one CSV → per-minute (day, minute, total, cold, cold_rate).
    Dataset: Huawei Public Cloud Trace 2025 – Region 1."""
    df = pd.read_csv(path)
    day_num = int(os.path.basename(path).split('_')[1].split('.')[0])
    df['time_sec'] = df['time'] % 86400
    df['minute']   = (df['time_sec'] // 60).astype(int)
    df['is_cold']  = (df['deployDependencyCost'] > 0).astype(int)
    pm = (
        df.groupby('minute')
          .agg(total=('requestID', 'count'), cold=('is_cold', 'sum'))
          .reset_index()
    )
    pm['cold_rate'] = pm['cold'] / pm['total']
    pm['day']       = day_num
    return pm[['day', 'minute', 'total', 'cold', 'cold_rate']]


def load_all_days(paths: list) -> pd.DataFrame:
    frames = []
    for p in sorted(paths):
        try:
            frames.append(load_day(p))
            print(f"  Loaded: {os.path.basename(p)}")
        except Exception as e:
            print(f"  WARNING: {p}: {e}")
    if not frames:
        raise ValueError("No valid CSV files loaded.")
    pm = pd.concat(frames, ignore_index=True)
    pm = pm.sort_values(['day', 'minute']).reset_index(drop=True)
    print(f"\nTotal: {pm['day'].nunique()} days, {len(pm):,} per-minute rows")
    return pm


# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def engineer_features(pm: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal, rolling, lag, trend, and target features.
    All rolling ops are scoped within each day to avoid midnight leakage.

    Uses an explicit per-day loop instead of groupby().apply() to avoid
    a pandas compatibility issue where newer versions drop the groupby key
    column ('day') from the group DataFrame, causing KeyError: 'day'.
    """
    pm = pm.copy().sort_values(['day', 'minute']).reset_index(drop=True)
    results = []

    for day_num in sorted(pm['day'].unique()):
        g = pm[pm['day'] == day_num].copy().reset_index(drop=True)

        g['hour']        = g['minute'] // 60
        g['is_business'] = ((g['hour'] >= 9) & (g['hour'] <= 17)).astype(int)
        g['day_of_week'] = (int(day_num) - 1) % 7

        for w in [5, 15, 60]:
            g[f'roll_mean_{w}'] = g['total'].rolling(w, min_periods=1).mean()
            g[f'roll_std_{w}']  = g['total'].rolling(w, min_periods=1).std().fillna(0)
            g[f'roll_cold_{w}'] = g['cold'].rolling(w, min_periods=1).mean()
        for lag in [1, 5, 15, 30]:
            g[f'lag_total_{lag}'] = g['total'].shift(lag).fillna(0)
            g[f'lag_cold_{lag}']  = g['cold'].shift(lag).fillna(0)
        g['trend_5']  = g['total'] - g['lag_total_5']
        g['trend_15'] = g['total'] - g['lag_total_15']

        day_mean = g['total'].mean()
        g['total_norm']        = g['total'] / (day_mean + 1e-6)
        g['burst_ratio']       = g['total'] / (g['roll_mean_60'] + 1e-6)
        g['cold_acceleration'] = g['cold_rate'] - g['cold_rate'].shift(15).fillna(0)

        for name, h in HORIZONS.items():
            g[name] = (
                g['cold_rate'].shift(-h).rolling(h, min_periods=1).mean()
                > COLD_THRESHOLD
            ).astype(int)

        results.append(g)

    return pd.concat(results, ignore_index=True).dropna().reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# 3. TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════

def split_by_day(pm_feat: pd.DataFrame):
    """
    Fixed research-grade split — explicitly defined, fully reproducible.
      Train : days 0–18  (19 days) — model learning
      Val   : days 19–24  (6 days) — hyperparameter tuning / early stopping
      Test  : days 25–30  (6 days) — held-out final evaluation, never touched during training
    """
    available = set(pm_feat['day'].unique())

    # Warn if any configured days are missing from the dataset
    for label, days in [('TRAIN', TRAIN_DAYS), ('VAL', VAL_DAYS), ('TEST', TEST_DAYS)]:
        missing = [d for d in days if d not in available]
        if missing:
            print(f"  WARNING: {label}_DAYS {missing} not found in dataset — skipping")

    train = pm_feat[pm_feat['day'].isin(TRAIN_DAYS)].copy()
    val   = pm_feat[pm_feat['day'].isin(VAL_DAYS)].copy()
    test  = pm_feat[pm_feat['day'].isin(TEST_DAYS)].copy()

    print(f"  Train : days {TRAIN_DAYS[0]}–{TRAIN_DAYS[-1]}  →  {len(train):,} rows  ({len(train)//1440} days)")
    print(f"  Val   : days {VAL_DAYS[0]}–{VAL_DAYS[-1]}      →  {len(val):,} rows  ({len(val)//1440} days)")
    print(f"  Test  : days {TEST_DAYS[0]}–{TEST_DAYS[-1]}     →  {len(test):,} rows  ({len(test)//1440} days)")

    if len(train) == 0: raise ValueError("Train set is empty — check TRAIN_DAYS vs your CSV filenames")
    if len(val)   == 0: raise ValueError("Val set is empty — check VAL_DAYS vs your CSV filenames")
    if len(test)  == 0: raise ValueError("Test set is empty — check TEST_DAYS vs your CSV filenames")

    return train, val, test


# ═══════════════════════════════════════════════════════════
# 4. SEQUENCE BUILDER  (no cross-day leakage)
# ═══════════════════════════════════════════════════════════

def make_sequences_by_day(pm_feat: pd.DataFrame, seq_data_sc: np.ndarray,
                           target_col: str, seq_len: int):
    """Build LSTM windows per day — no window ever crosses midnight."""
    Xs, ys = [], []
    pos = 0
    for day, group in pm_feat.groupby('day'):
        n          = len(group)
        day_seq    = seq_data_sc[pos:pos + n]
        day_labels = group[target_col].values
        for i in range(seq_len, n):
            Xs.append(day_seq[i - seq_len:i])
            ys.append(day_labels[i])
        pos += n
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def align_tabular(df: pd.DataFrame, X_tab: np.ndarray, seq_len: int):
    """Drop the first seq_len rows per day to align with sequence outputs."""
    rows = []
    pos  = 0
    for day, group in df.groupby('day'):
        n = len(group)
        for i in range(seq_len, n):
            rows.append(X_tab[pos + i])
        pos += n
    return np.array(rows)


# ═══════════════════════════════════════════════════════════
# 5. PYTORCH BIDIRECTIONAL LSTM
# ═══════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM cold-start predictor (PyTorch).

    Architecture:
        Input  (batch, seq_len, n_features)
        → BiLSTM(hidden=64, layers=1)  — smaller to reduce overfitting
        → last hidden state [fwd || bwd] = 128 dims
        → BatchNorm1d(128)
        → Linear(128→32) → ReLU → Dropout(0.3)
        → Linear(32→1) → Sigmoid

    Sized for ~15K–22K training sequences. Use hidden=128, num_layers=2
    only if you have 100K+ sequences.
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,   # no inter-layer dropout with single layer
        )
        self.bn   = nn.BatchNorm1d(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        last    = out[:, -1, :]      # last timestep
        last    = self.bn(last)
        return self.head(last).squeeze(1)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val:   np.ndarray):
    """Train BiLSTM with early stopping on validation AUC."""
    model     = BiLSTMClassifier(X_train.shape[2]).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # patience=3: drop LR quickly when val plateaus; factor=0.5: halve it
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5,
                                  min_lr=1e-5)
    criterion = nn.BCELoss()

    print(f"\nBiLSTM parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training on: {DEVICE}")

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE)

    history       = {'train_loss': [], 'val_auc': []}
    best_val_auc  = 0.0
    best_weights  = None
    patience_left = PATIENCE

    for epoch in range(1, LSTM_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(Xb)
        avg_loss = total_loss / len(y_train)

        # Validate
        model.eval()
        val_probs = []
        with torch.no_grad():
            for Xb, _ in val_dl:
                val_probs.append(model(Xb.to(DEVICE)).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_auc   = roc_auc_score(y_val, val_probs)

        scheduler.step(val_auc)
        history['train_loss'].append(avg_loss)
        history['val_auc'].append(val_auc)

        # Smooth AUC over last 3 epochs to reduce noise-driven early stopping
        smooth_auc = np.mean(history['val_auc'][-3:])

        print(f"Epoch {epoch:3d}/{LSTM_EPOCHS}  "
              f"loss={avg_loss:.4f}  val_AUC={val_auc:.4f}  "
              f"smooth={smooth_auc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping uses smoothed AUC to avoid stopping on a single noisy dip
        if smooth_auc > best_val_auc:
            best_val_auc  = smooth_auc
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} "
                      f"(best smoothed val_AUC={best_val_auc:.4f})")
                break

    model.load_state_dict(best_weights)
    return model, history


def predict_lstm(model: BiLSTMClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X)), batch_size=BATCH_SIZE)
    probs = []
    with torch.no_grad():
        for (Xb,) in dl:
            probs.append(model(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(probs)


# ═══════════════════════════════════════════════════════════
# 6. ADAPTIVE THRESHOLD CONTROLLER
# ═══════════════════════════════════════════════════════════

class AdaptiveThresholdController:
    """
    Online controller: adjusts decision threshold every step based on
    rolling false-positive / false-negative rates.
    - FP rate too high → raise threshold (reduce resource waste)
    - FN rate too high → lower threshold (catch more cold starts)
    """
    def __init__(self, base_threshold=0.5, window=30, alpha=0.03,
                 fp_limit=0.25, fn_limit=0.25):
        self.threshold = base_threshold
        self.window    = window
        self.alpha     = alpha
        self.fp_limit  = fp_limit
        self.fn_limit  = fn_limit
        self._history  = []

    def decide(self, prob: float) -> int:
        return int(prob >= self.threshold)

    def update(self, pred: int, actual: int):
        self._history.append((pred, actual))
        if len(self._history) > self.window:
            self._history.pop(0)
        if len(self._history) < 10:
            return
        preds, actuals = zip(*self._history)
        n_neg   = max(1, sum(a == 0 for a in actuals))
        n_pos   = max(1, sum(a == 1 for a in actuals))
        fp_rate = sum(p==1 and a==0 for p,a in zip(preds,actuals)) / n_neg
        fn_rate = sum(p==0 and a==1 for p,a in zip(preds,actuals)) / n_pos
        if fp_rate > self.fp_limit:
            self.threshold = min(0.85, self.threshold + self.alpha)
        elif fn_rate > self.fn_limit:
            self.threshold = max(0.15, self.threshold - self.alpha)


# ═══════════════════════════════════════════════════════════
# 7. SIMULATION
# ═══════════════════════════════════════════════════════════

def simulate(test_data: pd.DataFrame, hybrid_probs: np.ndarray,
             y_test: np.ndarray):
    """Compare No Warming / Fixed (0.5) / Adaptive on the test day."""
    controller = AdaptiveThresholdController()
    rows_adap, rows_fixed, rows_none = [], [], []

    for prob, actual, row in zip(hybrid_probs, y_test, test_data.itertuples()):
        pred_a = controller.decide(prob)
        cold_a = 0 if (pred_a == 1 and actual == 1) else int(actual == 1)
        controller.update(pred_a, actual)

        pred_f = int(prob >= 0.5)
        cold_f = 0 if (pred_f == 1 and actual == 1) else int(actual == 1)

        rows_adap.append({'day': row.day, 'minute': row.minute,
                          'threshold': controller.threshold,
                          'warmed': pred_a, 'cold_start': cold_a, 'prob': prob})
        rows_fixed.append({'warmed': pred_f, 'cold_start': cold_f})
        rows_none.append({'cold_start': int(actual == 1)})

    adap  = pd.DataFrame(rows_adap)
    fixed = pd.DataFrame(rows_fixed)
    none_ = pd.DataFrame(rows_none)

    summary = pd.DataFrame({
        'Strategy': ['No Warming (Baseline)',
                     'Fixed Threshold (0.5)',
                     'Adaptive (Proposed)'],
        'Cold Start Rate': [none_['cold_start'].mean(),
                            fixed['cold_start'].mean(),
                            adap['cold_start'].mean()],
        'Warmings': [0, int(fixed['warmed'].sum()), int(adap['warmed'].sum())],
    })
    summary['Reduction vs Baseline'] = (
        1 - summary['Cold Start Rate'] / summary['Cold Start Rate'].iloc[0]
    )
    return summary, adap, fixed, none_


# ═══════════════════════════════════════════════════════════
# 8. VISUALISATION
# ═══════════════════════════════════════════════════════════

def plot_eda(pm: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('31-Day Huawei Public Cloud Trace 2025 (Region 1) — EDA',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, :2])
    for day, grp in pm.groupby('day'):
        ax.plot(grp['minute']/60 + (day-1)*24, grp['total'], lw=0.6, alpha=0.7)
    ax.set_xlabel('Hour (continuous, 31 days)'); ax.set_ylabel('Invocations/min')
    ax.set_title('Invocation Volume — All 31 Days'); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, :2])
    for day, grp in pm.groupby('day'):
        ax.plot(grp['minute']/60 + (day-1)*24, grp['cold_rate'], lw=0.6, alpha=0.7)
    ax.axhline(pm['cold_rate'].mean(), ls='--', color='red',
               label=f"Mean={pm['cold_rate'].mean():.1%}")
    ax.set_xlabel('Hour (continuous)'); ax.set_ylabel('Cold Start Rate')
    ax.set_title('Cold Start Rate — All 31 Days'); ax.legend(); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    daily = pm.groupby('day').apply(
        lambda g: g['cold'].sum() / g['total'].sum()).reset_index()
    daily.columns = ['day', 'cold_rate']
    ax.bar(daily['day'], daily['cold_rate']*100, color='steelblue', alpha=0.85)
    ax.set_xlabel('Day'); ax.set_ylabel('Cold Start Rate (%)')
    ax.set_title('Per-Day Cold Start Rate'); ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 2])
    pm2 = pm.copy(); pm2['hour'] = pm2['minute'] // 60
    hourly = pm2.groupby('hour')['cold_rate'].mean()
    ax.bar(hourly.index, hourly.values*100, color='tomato', alpha=0.75)
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Avg Cold Start Rate (%)')
    ax.set_title('Hourly Cold Start Pattern (31-day avg)'); ax.grid(alpha=0.3, axis='y')

    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"EDA plot → {out_path}")


def plot_model_comparison(y_test, lstm_probs, rf_probs, hybrid_probs,
                          history, rf_importances, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Hybrid BiLSTM + RandomForest — 31-Day Performance\nHuawei Public Cloud Trace 2025 (Region 1)',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(history['val_auc'], color='tomato', lw=1.5, label='Val AUC')
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
    ax.set_title('LSTM Validation AUC'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for probs, label, color in [
        (lstm_probs,   f'LSTM   AUC={roc_auc_score(y_test,lstm_probs):.3f}',   'steelblue'),
        (rf_probs,     f'RF     AUC={roc_auc_score(y_test,rf_probs):.3f}',     'tomato'),
        (hybrid_probs, f'Hybrid AUC={roc_auc_score(y_test,hybrid_probs):.3f}', 'mediumseagreen'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=label, lw=2, color=color)
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    names = ['LSTM', 'RF', 'Hybrid']
    aucs  = [roc_auc_score(y_test, p) for p in [lstm_probs, rf_probs, hybrid_probs]]
    f1s   = [f1_score(y_test, (p>=0.5).astype(int)) for p in [lstm_probs, rf_probs, hybrid_probs]]
    x = np.arange(3); w = 0.35
    ax.bar(x-w/2, aucs, w, label='AUC', color='steelblue', alpha=0.85)
    ax.bar(x+w/2, f1s,  w, label='F1',  color='tomato',    alpha=0.85)
    for i,(a,f) in enumerate(zip(aucs,f1s)):
        ax.text(i-w/2, a+0.005, f'{a:.3f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(i+w/2, f+0.005, f'{f:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
    ax.set_title('AUC & F1'); ax.legend(); ax.grid(alpha=0.3, axis='y')

    ax = axes[3]
    idx = np.argsort(rf_importances)[-10:]
    ax.barh([TABULAR_FEATURES[i] for i in idx],
            rf_importances[idx], color='steelblue', alpha=0.85)
    ax.set_xlabel('Importance'); ax.set_title('Top 10 RF Feature Importances')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Model comparison plot → {out_path}")


def plot_multi_horizon(test_df, scaler_tab, rf, scaler_seq, lstm_model, out_path):
    labels  = ['1-min', '5-min', '15-min']
    results = {m: [] for m in ['LSTM', 'RF', 'Hybrid']}
    seq_test_sc = scaler_seq.transform(test_df[SEQUENCE_FEATURES].values)

    for h_col in HORIZONS.keys():
        X_seq_h, y_h = make_sequences_by_day(
            test_df, seq_test_sc, h_col, SEQUENCE_LEN)
        X_tab_h = align_tabular(
            test_df,
            scaler_tab.transform(test_df[TABULAR_FEATURES].values),
            SEQUENCE_LEN)
        lstm_p = predict_lstm(lstm_model, X_seq_h)
        rf_p   = rf.predict_proba(X_tab_h)[:, 1]
        hyb_p  = 0.5 * lstm_p + 0.5 * rf_p
        results['LSTM'].append(roc_auc_score(y_h, lstm_p))
        results['RF'].append(roc_auc_score(y_h, rf_p))
        results['Hybrid'].append(roc_auc_score(y_h, hyb_p))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Multi-Horizon Prediction AUC (Test Days 25–30)\nHuawei Public Cloud Trace 2025 (Region 1) — 31 Days',
                 fontsize=13, fontweight='bold')
    x = np.arange(3); w = 0.25
    for i, (model, color) in enumerate(
            zip(['LSTM','RF','Hybrid'],
                ['steelblue','tomato','mediumseagreen'])):
        bars = ax.bar(x+i*w, results[model], w, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, results[model]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x+w); ax.set_xticklabels(labels)
    ax.set_ylabel('AUC'); ax.set_ylim(0.5, 1.05)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Multi-horizon plot → {out_path}")


def plot_simulation(summary, adap_df, fixed_df, none_df, out_path):
    minutes = np.arange(len(adap_df)); w = 15
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Simulation Results — Test Days 25–30\nHuawei Public Cloud Trace 2025 (Region 1)', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(minutes, pd.Series(none_df['cold_start'].values).rolling(w).mean(),
            label='No Warming', color='tomato', lw=1.5)
    ax.plot(minutes, pd.Series(fixed_df['cold_start'].values).rolling(w).mean(),
            label='Fixed (0.5)', color='gold', lw=1.5)
    ax.plot(minutes, adap_df['cold_start'].rolling(w).mean(),
            label='Adaptive+Hybrid', color='mediumseagreen', lw=2)
    ax.set_xlabel('Minute'); ax.set_ylabel(f'Rolling Cold Start Rate (w={w})')
    ax.set_title('Cold Start Rate Over Time'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(minutes, adap_df['prob'], color='steelblue', lw=0.8, alpha=0.7, label='Hybrid Prob')
    ax.plot(minutes, adap_df['threshold'], color='darkorange', lw=1.5, ls='--', label='Threshold')
    ax.fill_between(minutes, adap_df['warmed']*0.9, alpha=0.12, color='green', label='Warming')
    ax.set_xlabel('Minute'); ax.set_title('Adaptive Threshold Controller')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    rates  = summary['Cold Start Rate'].values * 100
    colors = ['tomato', 'gold', 'mediumseagreen']
    bars   = ax.bar(summary['Strategy'], rates, color=colors, edgecolor='white', width=0.5)
    for bar, rate, red in zip(bars, rates, summary['Reduction vs Baseline'].values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{rate:.1f}%\n({red:.0%} ↓)', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Cold Start Rate (%)'); ax.set_title('Strategy Summary')
    ax.set_xticklabels(summary['Strategy'], rotation=12, ha='right')
    ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, rates.max()*1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Simulation plot → {out_path}")


# ═══════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline(csv_paths, out_dir='results', model_dir='models'):
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — LOADING DATA")
    print("="*60)
    pm = load_all_days(csv_paths)

    # ── 2. EDA ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — EDA")
    print("="*60)
    print(f"Total invocations : {pm['total'].sum():,}")
    print(f"Overall cold rate : {pm['cold'].sum()/pm['total'].sum():.2%}")
    plot_eda(pm, os.path.join(out_dir, 'eda_31day.png'))

    # ── 3. Features ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — FEATURE ENGINEERING")
    print("="*60)
    pm_feat = engineer_features(pm)
    print(f"Feature matrix: {len(pm_feat):,} rows × "
          f"{len(TABULAR_FEATURES)} tabular + {len(SEQUENCE_FEATURES)} seq features")

    # ── 4. Split ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — CROSS-DAY SPLIT  (fixed research-grade)")
    print("="*60)
    train_df, val_df, test_df = split_by_day(pm_feat)
    train_days = TRAIN_DAYS
    val_days   = VAL_DAYS
    test_days  = TEST_DAYS

    # ── 5. Prepare arrays ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — PREPARING ARRAYS")
    print("="*60)
    scaler_tab   = StandardScaler()
    X_train_tab  = scaler_tab.fit_transform(train_df[TABULAR_FEATURES].values)
    X_val_tab    = scaler_tab.transform(val_df[TABULAR_FEATURES].values)
    y_train      = train_df[PRIMARY_HORIZON].values
    y_val        = val_df[PRIMARY_HORIZON].values

    scaler_seq   = StandardScaler()
    seq_train_sc = scaler_seq.fit_transform(train_df[SEQUENCE_FEATURES].values)
    seq_val_sc   = scaler_seq.transform(val_df[SEQUENCE_FEATURES].values)
    seq_test_sc  = scaler_seq.transform(test_df[SEQUENCE_FEATURES].values)

    X_seq_train, y_seq_train = make_sequences_by_day(
        train_df, seq_train_sc, PRIMARY_HORIZON, SEQUENCE_LEN)
    X_seq_val,   y_seq_val   = make_sequences_by_day(
        val_df,   seq_val_sc,   PRIMARY_HORIZON, SEQUENCE_LEN)
    X_seq_test,  y_seq_test  = make_sequences_by_day(
        test_df,  seq_test_sc,  PRIMARY_HORIZON, SEQUENCE_LEN)

    X_val_tab_aligned  = align_tabular(val_df,  X_val_tab,  SEQUENCE_LEN)
    X_test_tab_aligned = align_tabular(
        test_df,
        scaler_tab.transform(test_df[TABULAR_FEATURES].values),
        SEQUENCE_LEN)

    print(f"LSTM — train:{X_seq_train.shape} val:{X_seq_val.shape} test:{X_seq_test.shape}")

    # ── 6. RandomForest ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — TRAINING RANDOM FOREST")
    print("="*60)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                random_state=42, n_jobs=-1,
                                class_weight='balanced')
    rf.fit(X_train_tab, y_train)
    rf_val_probs  = rf.predict_proba(X_val_tab_aligned)[:, 1]
    rf_test_probs = rf.predict_proba(X_test_tab_aligned)[:, 1]
    print(f"RF val  AUC: {roc_auc_score(y_seq_val,  rf_val_probs):.4f}")
    print(f"RF test AUC: {roc_auc_score(y_seq_test, rf_test_probs):.4f}")

    # ── 7. BiLSTM ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7 — TRAINING BIDIRECTIONAL LSTM (PyTorch)")
    print("="*60)
    lstm_model, history = train_lstm(
        X_seq_train, y_seq_train, X_seq_val, y_seq_val)
    lstm_val_probs  = predict_lstm(lstm_model, X_seq_val)
    lstm_test_probs = predict_lstm(lstm_model, X_seq_test)
    print(f"LSTM val  AUC: {roc_auc_score(y_seq_val,  lstm_val_probs):.4f}")
    print(f"LSTM test AUC: {roc_auc_score(y_seq_test, lstm_test_probs):.4f}")

    # ── 8. Ensemble ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8 — CONFIDENCE-WEIGHTED HYBRID ENSEMBLE")
    print("="*60)
    w_lstm  = roc_auc_score(y_seq_val, lstm_val_probs)
    w_rf    = roc_auc_score(y_seq_val, rf_val_probs)
    alpha_l = w_lstm / (w_lstm + w_rf)
    alpha_r = w_rf   / (w_lstm + w_rf)
    print(f"Ensemble weights → LSTM: {alpha_l:.3f}  RF: {alpha_r:.3f}")

    hybrid_probs = alpha_l * lstm_test_probs + alpha_r * rf_test_probs

    print("\n=== Final Results (Test Day) ===")
    for name, probs in [('LSTM',   lstm_test_probs),
                         ('RF',     rf_test_probs),
                         ('Hybrid', hybrid_probs)]:
        preds = (probs >= 0.5).astype(int)
        print(f"  {name:8s}: AUC={roc_auc_score(y_seq_test,probs):.4f}  "
              f"F1={f1_score(y_seq_test,preds):.4f}  "
              f"P={precision_score(y_seq_test,preds):.4f}  "
              f"R={recall_score(y_seq_test,preds):.4f}")

    # ── 9. Save models ───────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 9 — SAVING MODELS")
    print("="*60)
    lstm_path = os.path.join(model_dir, 'lstm_model.pt')
    rf_path   = os.path.join(model_dir, 'rf_hybrid.pkl')
    torch.save({'model_state': lstm_model.state_dict(),
                'n_features':  len(SEQUENCE_FEATURES)}, lstm_path)
    with open(rf_path, 'wb') as f:
        pickle.dump({
            'rf': rf,
            'scaler_tab': scaler_tab,
            'scaler_seq': scaler_seq,
            'tabular_features':  TABULAR_FEATURES,
            'sequence_features': SEQUENCE_FEATURES,
            'sequence_len':      SEQUENCE_LEN,
            'ensemble_weights':  {'lstm': alpha_l, 'rf': alpha_r},
            'train_days': TRAIN_DAYS,
            'val_days':   VAL_DAYS,
            'test_days':  TEST_DAYS,
        }, f)
    print(f"Saved: {lstm_path}")
    print(f"Saved: {rf_path}")

    # ── 10. Plots ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 10 — PLOTS")
    print("="*60)
    plot_model_comparison(
        y_seq_test, lstm_test_probs, rf_test_probs, hybrid_probs,
        history, rf.feature_importances_,
        os.path.join(out_dir, 'model_comparison.png'))

    plot_multi_horizon(
        test_df, scaler_tab, rf, scaler_seq, lstm_model,
        os.path.join(out_dir, 'multi_horizon_results.png'))

    # ── 11. Simulation ───────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 11 — SIMULATION")
    print("="*60)
    test_rows = []
    for day, grp in test_df.groupby('day'):
        test_rows.extend(grp.iloc[SEQUENCE_LEN:].to_dict('records'))
    test_data_aligned = pd.DataFrame(test_rows).reset_index(drop=True)

    summary, adap_df, fixed_df, none_df = simulate(
        test_data_aligned, hybrid_probs, y_seq_test)
    print("\n=== Simulation Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(out_dir, 'simulation_summary.csv'), index=False)
    plot_simulation(summary, adap_df, fixed_df, none_df,
                    os.path.join(out_dir, 'simulation.png'))

    print("\n" + "="*60)
    print(f"DONE — results in {out_dir}/   models in {model_dir}/")
    print("="*60)
    return summary


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='31-Day Cold Start Mitigation — PyTorch Edition')
    parser.add_argument('--dir', type=str, default=None,
                        help='Folder containing day_01.csv … day_14.csv')
    parser.add_argument('csvs', nargs='*', help='Explicit CSV paths')
    parser.add_argument('--out',    type=str, default='results')
    parser.add_argument('--models', type=str, default='models')
    args = parser.parse_args()

    if args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, 'day_*.csv')))
    elif args.csvs:
        paths = args.csvs
    else:
        paths = sorted(glob.glob('day_*.csv'))

    if not paths:
        print("ERROR: No CSV files found.")
        sys.exit(1)

    print(f"Found {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    run_pipeline(paths, out_dir=args.out, model_dir=args.models)