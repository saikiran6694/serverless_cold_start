"""
Proactive Cold Start Mitigation in Serverless Environments
==========================================================
Full 31-Day Pipeline — EDA, Feature Engineering, BiLSTM + RandomForest Hybrid, Adaptive Threshold Controller, Simulation, Multi-Horizon Evaluation
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
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)
from data_loader.loader import load_all_days
from feature_engineer.feature_engineering import engineer_features, split_by_day, make_sequences_by_day, align_tabular
from visualization.plots import plot_eda, plot_model_comparison, plot_multi_horizon, plot_simulation
from config.config import *
from models.bilstm_model import train_lstm, predict_lstm
from models.simulator import simulate


# ═══════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline(csv_paths, out_dir='results', model_dir='generated_models'):
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
    print(f"DONE — results in {out_dir}/   generated_models in {model_dir}/")
    print("="*60)
    return summary



# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='31-Day Cold Start Mitigation')
    parser.add_argument('--dir', type=str, default=None,
                        help='Folder containing day_01.csv … day_14.csv')
    parser.add_argument('csvs', nargs='*', help='Explicit CSV paths')
    parser.add_argument('--out',    type=str, default='results')
    parser.add_argument('--models', type=str, default='generated_models')
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