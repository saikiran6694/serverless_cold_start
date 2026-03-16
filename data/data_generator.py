"""
Synthetic Azure Functions trace data generator.
Simulates 14-day production invocation traces with realistic patterns:
- Daily/weekly cycles
- Bursty traffic
- Business-hour peaks
- Long-tail idle periods
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_azure_traces(
    n_days: int = 14,
    n_functions: int = 50,
    seed: int = 42,
    resolution_minutes: int = 1
) -> pd.DataFrame:
    """
    Generate synthetic Azure Functions invocation traces.
    Returns per-minute invocation counts per function.
    """
    np.random.seed(seed)
    total_minutes = n_days * 24 * 60
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(total_minutes)]

    records = []
    function_types = ["api", "webhook", "batch", "event", "scheduled"]
    http_methods = ["GET", "POST", "PUT", "DELETE"]
    payload_sizes = [128, 512, 1024, 4096, 16384]  # bytes

    for fn_id in range(n_functions):
        fn_type = np.random.choice(function_types)
        base_load = np.random.uniform(0.5, 20.0)       # avg invocations/min
        burstiness = np.random.uniform(1.5, 8.0)       # burst multiplier
        has_daily_cycle = np.random.random() > 0.2
        has_weekly_cycle = np.random.random() > 0.4
        is_scheduled = fn_type == "scheduled"

        invocations = []
        for t, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.weekday()    # 0=Mon, 6=Sun
            minute = ts.minute

            # Base rate
            rate = base_load

            # Daily cycle (business hours peak)
            if has_daily_cycle:
                if 9 <= hour <= 17:
                    rate *= np.random.uniform(2.0, 4.0)
                elif 18 <= hour <= 22:
                    rate *= np.random.uniform(1.2, 2.0)
                elif 0 <= hour <= 6:
                    rate *= np.random.uniform(0.05, 0.3)

            # Weekly cycle (weekday higher)
            if has_weekly_cycle:
                if dow < 5:   # weekday
                    rate *= np.random.uniform(1.5, 2.5)
                else:         # weekend
                    rate *= np.random.uniform(0.3, 0.8)

            # Scheduled functions: spike at fixed intervals
            if is_scheduled:
                interval = np.random.choice([5, 10, 15, 30, 60])
                if minute % interval == 0:
                    rate *= np.random.uniform(5.0, 15.0)
                else:
                    rate *= 0.0

            # Random burst events (5% chance)
            if np.random.random() < 0.05:
                rate *= burstiness * np.random.uniform(0.5, 2.0)

            # Idle periods (10% chance per minute for low-traffic functions)
            if base_load < 2.0 and np.random.random() < 0.15:
                rate = 0.0
            # General idle: even busy functions sometimes go quiet
            if np.random.random() < 0.03:
                rate = 0.0

            count = np.random.poisson(max(0, rate))
            invocations.append(count)

        # Build per-minute records for this function
        fn_records = pd.DataFrame({
            "timestamp": timestamps,
            "function_id": fn_id,
            "function_type": fn_type,
            "invocations": invocations,
            "http_method": np.random.choice(http_methods, total_minutes, p=[0.5, 0.3, 0.1, 0.1]),
            "avg_payload_bytes": np.random.choice(payload_sizes, total_minutes),
            "avg_execution_ms": np.random.lognormal(
                mean=np.log(base_load * 20 + 50), sigma=0.5, size=total_minutes
            ).clip(10, 5000),
            "error_rate": np.random.beta(0.5, 10, total_minutes).clip(0, 0.3),
        })
        records.append(fn_records)

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["timestamp", "function_id"]).reset_index(drop=True)
    return df


def aggregate_global(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all functions into global per-minute invocation series."""
    global_df = df.groupby("timestamp").agg(
        total_invocations=("invocations", "sum"),
        active_functions=("invocations", lambda x: (x > 0).sum()),
        avg_execution_ms=("avg_execution_ms", "mean"),
        avg_error_rate=("error_rate", "mean"),
    ).reset_index()
    global_df = global_df.sort_values("timestamp").reset_index(drop=True)
    return global_df


if __name__ == "__main__":
    print("Generating synthetic Azure Functions traces...")
    df = generate_azure_traces(n_days=14, n_functions=50)
    global_df = aggregate_global(df)
    print(f"Generated {len(df):,} per-function records")
    print(f"Global series: {len(global_df):,} minutes")
    print(f"Total invocations: {df['invocations'].sum():,}")
    print(global_df.head(10))