import pandas as pd
import numpy as np
from config.config import (HORIZONS, COLD_THRESHOLD, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)


def engineer_features(pm: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal, rolling, lag, trend, and target features.
    All rolling ops are scoped within each day to avoid midnight leakage.
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
            print(f"  WARNING   :    {label}_DAYS {missing} not found in dataset — skipping")

    train = pm_feat[pm_feat['day'].isin(TRAIN_DAYS)].copy()
    val   = pm_feat[pm_feat['day'].isin(VAL_DAYS)].copy()
    test  = pm_feat[pm_feat['day'].isin(TEST_DAYS)].copy()

    print(f"  Train     :    days {TRAIN_DAYS[0]}–{TRAIN_DAYS[-1]}  →  {len(train):,} rows  ({len(train)//1440} days)")
    print(f"  Val       :    days {VAL_DAYS[0]}–{VAL_DAYS[-1]}      →  {len(val):,} rows  ({len(val)//1440} days)")
    print(f"  Test      :    days {TEST_DAYS[0]}–{TEST_DAYS[-1]}     →  {len(test):,} rows  ({len(test)//1440} days)")

    if len(train) == 0: raise ValueError("Train set is empty — check TRAIN_DAYS vs your CSV filenames")
    if len(val)   == 0: raise ValueError("Val set is empty — check VAL_DAYS vs your CSV filenames")
    if len(test)  == 0: raise ValueError("Test set is empty — check TEST_DAYS vs your CSV filenames")

    return train, val, test

# ══════════════════════════════════════════════════════════
# 4. SEQUENCE PREPARATION
# ══════════════════════════════════════════════════════════
def make_sequences_by_day(pm_feat: pd.DataFrame, seq_data_sc: np.ndarray,
                           target_col: str, seq_len: int):
    """
    Build LSTM windows per day — no window ever crosses midnight.
    Returns:
      Xs: (num_samples, seq_len, num_features) — 3D array of input sequences
      ys: (num_samples,) — 1D array of binary labels for each sequence
    """
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
    """
    Drop the first seq_len rows per day to align with sequence outputs.
    Returns:
      rows: (num_samples, num_tab_features) — 2D array of tabular features aligned with sequences
    """
    rows = []
    pos  = 0
    for day, group in df.groupby('day'):
        n = len(group)
        for i in range(seq_len, n):
            rows.append(X_tab[pos + i])
        pos += n
    return np.array(rows)