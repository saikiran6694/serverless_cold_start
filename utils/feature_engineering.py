"""
Feature engineering for cold start prediction.
Extracts temporal, statistical, and contextual features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FEATURE_CONFIG
from data.data_generator import generate_azure_traces, aggregate_global


def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add time-based features from timestamp."""
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_business_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 17) & (ts.dt.dayofweek < 5)).astype(int)
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    return df


def add_rolling_features(df: pd.DataFrame, value_col: str = "total_invocations",
                          windows: List[int] = None) -> pd.DataFrame:
    """Add rolling statistical features."""
    if windows is None:
        windows = FEATURE_CONFIG["rolling_windows"]
    df = df.copy()
    for w in windows:
        df[f"roll_mean_{w}m"] = df[value_col].rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}m"] = df[value_col].rolling(w, min_periods=1).std().fillna(0)
        df[f"roll_max_{w}m"] = df[value_col].rolling(w, min_periods=1).max()
        df[f"roll_min_{w}m"] = df[value_col].rolling(w, min_periods=1).min()
    # Trend: difference of recent means
    if len(windows) >= 2:
        df["short_long_ratio"] = (df[f"roll_mean_{windows[0]}m"] /
                                   (df[f"roll_mean_{windows[-1]}m"] + 1e-9))
    # Lag features
    for lag in [1, 5, 10, 30, 60]:
        df[f"lag_{lag}m"] = df[value_col].shift(lag).fillna(0)
    return df


def add_burst_features(df: pd.DataFrame, value_col: str = "total_invocations") -> pd.DataFrame:
    """Detect burst characteristics."""
    df = df.copy()
    # Z-score relative to rolling baseline
    baseline = df[value_col].rolling(60, min_periods=1).mean()
    std = df[value_col].rolling(60, min_periods=1).std().fillna(1).replace(0, 1)
    df["zscore_60m"] = (df[value_col] - baseline) / std
    df["is_burst"] = (df["zscore_60m"] > 2.0).astype(int)
    # Time since last invocation (idle detection)
    df["time_since_last"] = 0
    last_active = 0
    for i, row in df.iterrows():
        if row[value_col] > 0:
            last_active = 0
        else:
            last_active += 1
        df.at[i, "time_since_last"] = last_active
    return df


def create_prediction_labels(df: pd.DataFrame, value_col: str = "total_invocations",
                               horizons: List[int] = None) -> pd.DataFrame:
    """
    Create binary labels: will there be a SPIKE (above rolling average) in the next H minutes?
    Also create regression targets (future invocation counts).
    """
    if horizons is None:
        horizons = FEATURE_CONFIG["prediction_horizons"]
    df = df.copy()
    # Compute rolling baseline for spike detection
    rolling_avg = df[value_col].rolling(30, min_periods=1).mean()
    for h in horizons:
        # Regression: exact count h minutes ahead
        future_val = df[value_col].shift(-h).fillna(0).astype(float)
        df[f"label_count_{h}m"] = future_val
        # Binary: future value is above 1.5x recent rolling average (invocation spike)
        future_baseline = rolling_avg.shift(-h).fillna(rolling_avg.mean())
        df[f"label_binary_{h}m"] = (future_val > future_baseline * 1.2).astype(int)
    return df


def build_feature_matrix(global_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: build feature matrix from global invocation series."""
    df = global_df.copy()
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_burst_features(df)
    df = create_prediction_labels(df)
    # Drop NaN rows (from rolling windows at the start)
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns() -> List[str]:
    """Return ordered list of feature column names used for ML models."""
    temporal = [
        "hour", "minute", "day_of_week", "week_of_year", "month",
        "is_weekend", "is_business_hour",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "minute_sin", "minute_cos"
    ]
    rolling = []
    for w in FEATURE_CONFIG["rolling_windows"]:
        rolling += [f"roll_mean_{w}m", f"roll_std_{w}m", f"roll_max_{w}m", f"roll_min_{w}m"]
    rolling += ["short_long_ratio"]
    lags = [f"lag_{lag}m" for lag in [1, 5, 10, 30, 60]]
    burst = ["zscore_60m", "is_burst", "time_since_last"]
    context = ["active_functions", "avg_execution_ms", "avg_error_rate"]
    return temporal + rolling + lags + burst + context


def train_val_test_split(df: pd.DataFrame, train_pct: float = 0.70,
                          val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split (no shuffle for time series)."""
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


if __name__ == "__main__":
    
    print("Building feature matrix...")
    raw = generate_azure_traces(n_days=14, n_functions=10)
    global_df = aggregate_global(raw)
    features = build_feature_matrix(global_df)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(features[get_feature_columns()].describe())