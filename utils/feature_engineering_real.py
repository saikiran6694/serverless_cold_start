"""
Additional feature engineering specific to the real Azure dataset.
Adds cold-start-aware features on top of the base feature matrix.
"""

import numpy as np
import pandas as pd


def add_cold_start_features(feat_df: pd.DataFrame,
                              global_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cold-start-specific columns from the global time series
    into the feature matrix, then engineer derived features.
    """
    feat_df = feat_df.copy()

    # Merge real cold start metrics if present in global_ts
    real_cols = ["cold_starts", "cold_start_rate", "avg_duration_ms", "p95_duration_ms"]
    available = [c for c in real_cols if c in global_ts.columns]

    if available:
        merge_df = global_ts[["timestamp"] + available].copy()
        merge_df["timestamp"] = pd.to_datetime(merge_df["timestamp"])
        feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
        feat_df = feat_df.merge(merge_df, on="timestamp", how="left")

    # Rolling cold start rate features
    if "cold_start_rate" in feat_df.columns:
        feat_df["roll_cs_rate_5m"]  = feat_df["cold_start_rate"].rolling(5,  min_periods=1).mean()
        feat_df["roll_cs_rate_15m"] = feat_df["cold_start_rate"].rolling(15, min_periods=1).mean()
        feat_df["roll_cs_rate_60m"] = feat_df["cold_start_rate"].rolling(60, min_periods=1).mean()
        # Trend: is cold start rate rising?
        feat_df["cs_rate_trend"] = feat_df["roll_cs_rate_5m"] - feat_df["roll_cs_rate_60m"]

    # Rolling duration features
    if "avg_duration_ms" in feat_df.columns:
        feat_df["roll_avg_dur_15m"] = feat_df["avg_duration_ms"].rolling(15, min_periods=1).mean()
        feat_df["roll_avg_dur_60m"] = feat_df["avg_duration_ms"].rolling(60, min_periods=1).mean()

    # Lag cold start rate
    if "cold_start_rate" in feat_df.columns:
        for lag in [1, 5, 10, 30]:
            feat_df[f"lag_cs_rate_{lag}m"] = feat_df["cold_start_rate"].shift(lag).fillna(0)

    feat_df = feat_df.fillna(0)
    return feat_df