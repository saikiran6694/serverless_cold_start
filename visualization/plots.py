import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from config.config import HORIZONS, SEQUENCE_FEATURES, TABULAR_FEATURES, SEQUENCE_LEN
from models.bilstm_model import predict_lstm
from feature_engineer.feature_engineering import make_sequences_by_day, align_tabular


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