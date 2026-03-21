"""
Real Azure Functions Data Loader
Handles large CSVs (350MB+) via chunked reading.

Schema: app, func, end_timestamp, duration
- end_timestamp : float — when the function finished.
                  Auto-detected as milliseconds (>1e11) or seconds.
                  If values look like a relative offset from a custom
                  epoch (e.g. the public Azure 2019 dataset uses
                  milliseconds since a trace-start epoch), they are
                  rebased to a real calendar date (2020-01-01 UTC) so
                  that temporal features (hour-of-day, day-of-week)
                  are meaningful.
- duration      : float (seconds)      — how long it ran
- start_timestamp is inferred as: end_timestamp_s - duration
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Iterator
from datetime import datetime, timezone


# ── tuneable ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 200_000     # rows per chunk
COLD_START_GAP  = 10 * 60    # seconds; gap > this → cold start (10-min TTL)
RESAMPLE_FREQ   = "1min"      # aggregation resolution

# Anchor epoch used when timestamps are relative offsets.
# 2020-01-06 00:00 UTC is a Monday, giving realistic business-hour patterns.
_ANCHOR_EPOCH_S = int(datetime(2020, 1, 6, tzinfo=timezone.utc).timestamp())


def _normalise_timestamps(series: pd.Series) -> pd.Series:
    """
    Convert end_timestamp to Unix seconds (float64).

    Handles four common formats found in public Azure Function datasets:

      1. Unix seconds  (median ~ 1.5e9..2e9)
         → use as-is.

      2. Unix milliseconds  (median > 1e11)
         → divide by 1000.

      3. Relative seconds from trace start  (max <= ~2.5e6, i.e. <= ~30 days)
         This is the format used by the bundled azure_traces.csv:
         values run from 0 to 1,209,600 (= 14 days in seconds).
         → add _ANCHOR_EPOCH_S to rebase onto a real calendar date.

      4. Relative milliseconds from trace start  (max > 2.5e6 but < 9.46e8)
         → divide by 1000, then add _ANCHOR_EPOCH_S.
    """
    s = series.astype("float64")
    median_val = s.median()
    max_val    = s.max()

    if median_val > 1e11:
        # Format 2: Unix milliseconds
        s = s / 1000.0

    elif median_val > 9.46e8:
        # Format 1: already Unix seconds — use as-is
        pass

    elif max_val <= 2_592_000:
        # Format 3: relative seconds (max <= 30 days worth of seconds).
        # The bundled azure_traces.csv falls here: max ~1,209,600 s = 14 days.
        s = s + _ANCHOR_EPOCH_S

    else:
        # Format 4: relative milliseconds
        s = (s / 1000.0) + _ANCHOR_EPOCH_S

    # Sanity-check: final values must be plausible Unix seconds (2000–2030)
    lo, hi = 9.46e8, 1.9e9
    out_of_range = (~s.between(lo, hi)).mean()
    if out_of_range > 0.05:
        import warnings
        warnings.warn(
            f"[data_loader_real] {out_of_range*100:.1f}% of timestamps are outside "
            f"the expected 2000–2030 range after normalisation. "
            f"median_raw={median_val:.3e}, max_raw={max_val:.3e}. "
            f"Check the CSV timestamp column.",
            RuntimeWarning,
            stacklevel=2,
        )

    return s


# ── 1. Chunked raw reader ─────────────────────────────────────────────────────
def iter_chunks(csv_path: str, chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield cleaned DataFrame chunks from the raw CSV."""
    dtypes = {
        "app":           "string",
        "func":          "string",
        "end_timestamp": "float64",
        "duration":      "float64",
    }
    _ts_normalised = False   # only probe once per file
    _ts_offset = 0.0

    for chunk in pd.read_csv(csv_path, dtype=dtypes, chunksize=chunk_size):
        # Drop bad rows
        chunk = chunk.dropna(subset=["end_timestamp", "duration"])
        chunk = chunk[chunk["duration"] >= 0]
        chunk = chunk[chunk["end_timestamp"] > 0]

        # Normalise timestamps (detect ms / relative offsets)
        chunk["end_timestamp"] = _normalise_timestamps(chunk["end_timestamp"])

        # Infer start
        chunk["start_timestamp"] = chunk["end_timestamp"] - chunk["duration"]
        chunk["start_dt"] = pd.to_datetime(chunk["start_timestamp"], unit="s", utc=True)
        chunk["end_dt"]   = pd.to_datetime(chunk["end_timestamp"],   unit="s", utc=True)
        yield chunk


# ── 2. Build per-function sorted invocation table ─────────────────────────────
def load_sorted_invocations(csv_path: str) -> pd.DataFrame:
    """
    Read full CSV in chunks, sort each function's invocations by start time.
    Returns a DataFrame with cold_start flag added.
    Memory-efficient: streams chunks, accumulates only necessary columns.
    """
    print(f"  Loading {csv_path} in chunks of {CHUNK_SIZE:,} rows …")
    parts = []
    total_rows = 0

    for i, chunk in enumerate(iter_chunks(csv_path)):
        parts.append(chunk[["app", "func", "start_timestamp", "start_dt",
                             "end_timestamp", "end_dt", "duration"]])
        total_rows += len(chunk)
        if (i + 1) % 5 == 0:
            print(f"    … read {total_rows:,} rows so far")

    print(f"  Total rows loaded: {total_rows:,}")
    df = pd.concat(parts, ignore_index=True)

    # Sort globally by start time
    df = df.sort_values("start_timestamp").reset_index(drop=True)
    return df


# ── 3. Infer cold starts ───────────────────────────────────────────────────────
def add_cold_start_flag(df: pd.DataFrame,
                         gap_threshold_s: float = COLD_START_GAP) -> pd.DataFrame:
    """
    For each (app, func) pair, flag an invocation as a cold start if:
      - it is the first invocation ever, OR
      - the gap since the previous invocation's END > gap_threshold_s
    This mirrors real serverless container TTL behaviour.
    """
    print(f"  Inferring cold starts (TTL gap = {gap_threshold_s/60:.0f} min) …")
    df = df.copy()
    df = df.sort_values(["func", "start_timestamp"])

    # Previous end time per function
    df["prev_end"] = df.groupby("func")["end_timestamp"].shift(1)
    df["gap_s"]    = df["start_timestamp"] - df["prev_end"]

    # Cold start = first call OR gap > threshold
    df["cold_start"] = df["prev_end"].isna() | (df["gap_s"] > gap_threshold_s)
    df["cold_start"] = df["cold_start"].astype(int)

    cold_rate = df["cold_start"].mean() * 100
    print(f"  Cold start rate in raw data: {cold_rate:.2f}%")
    return df


# ── 4. Global per-minute aggregation ─────────────────────────────────────────
def build_global_timeseries(df: pd.DataFrame,
                              freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Aggregate all invocations into a per-minute global time series.
    Returns columns:
      timestamp, total_invocations, cold_starts, cold_start_rate,
      active_functions, avg_duration_ms, p95_duration_ms, total_duration_s
    """
    print(f"  Building global {freq} time series …")
    df2 = df.set_index("start_dt").sort_index()

    agg = df2.resample(freq).agg(
        total_invocations  = ("duration",    "count"),
        cold_starts        = ("cold_start",  "sum"),
        active_functions   = ("func",        "nunique"),
        avg_duration_ms    = ("duration",    lambda x: x.mean() * 1000),
        p95_duration_ms    = ("duration",    lambda x: np.percentile(x, 95) * 1000
                                             if len(x) > 0 else 0),
        total_duration_s   = ("duration",    "sum"),
    ).reset_index()

    agg = agg.rename(columns={"start_dt": "timestamp"})
    agg["cold_start_rate"] = agg["cold_starts"] / agg["total_invocations"].replace(0, np.nan)
    agg["cold_start_rate"] = agg["cold_start_rate"].fillna(0)

    # Remove timezone for downstream compatibility
    agg["timestamp"] = agg["timestamp"].dt.tz_localize(None)

    print(f"  Time series: {len(agg):,} minutes "
          f"({agg['timestamp'].min()} → {agg['timestamp'].max()})")
    print(f"  Total invocations : {agg['total_invocations'].sum():,}")
    print(f"  Total cold starts : {agg['cold_starts'].sum():,}")
    print(f"  Avg cold start rate: {agg['cold_start_rate'].mean()*100:.2f}%")
    return agg


# ── 5. Per-function aggregation (for function-level analysis) ──────────────────
def build_function_timeseries(df: pd.DataFrame,
                               top_n: int = 20,
                               freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Build per-minute time series for the top_n busiest functions.
    Useful for function-level cold start analysis.
    """
    print(f"  Building per-function time series (top {top_n} functions) …")
    top_funcs = df["func"].value_counts().head(top_n).index.tolist()
    df_top = df[df["func"].isin(top_funcs)].copy()
    df_top = df_top.set_index("start_dt").sort_index()

    records = []
    for func_id in top_funcs:
        sub = df_top[df_top["func"] == func_id]
        agg = sub.resample(freq).agg(
            invocations = ("duration", "count"),
            cold_starts = ("cold_start", "sum"),
            avg_duration_ms = ("duration", lambda x: x.mean() * 1000),
        ).reset_index()
        agg["func"] = func_id
        agg = agg.rename(columns={"start_dt": "timestamp"})
        agg["timestamp"] = agg["timestamp"].dt.tz_localize(None)
        records.append(agg)

    result = pd.concat(records, ignore_index=True)
    return result


# ── 6. Quick dataset summary ───────────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame) -> dict:
    """Print and return key dataset statistics."""
    n_apps   = df["app"].nunique()
    n_funcs  = df["func"].nunique()
    n_rows   = len(df)
    dur_span = (df["start_timestamp"].max() - df["start_timestamp"].min()) / 86400

    # Use start_dt (already normalised + tz-aware) for human-readable dates.
    # Falling back to raw start_timestamp only if start_dt column is absent.
    if "start_dt" in df.columns:
        start_date = str(df["start_dt"].min())
        end_date   = str(df["start_dt"].max())
    else:
        start_date = str(pd.to_datetime(df["start_timestamp"].min(), unit="s", utc=True))
        end_date   = str(pd.to_datetime(df["start_timestamp"].max(), unit="s", utc=True))

    summary = {
        "total_invocations":   n_rows,
        "unique_apps":         n_apps,
        "unique_functions":    n_funcs,
        "trace_duration_days": round(dur_span, 2),
        "cold_start_rate_pct": round(df["cold_start"].mean() * 100, 2),
        "median_duration_s":   round(df["duration"].median(), 4),
        "p95_duration_s":      round(df["duration"].quantile(0.95), 4),
        "p99_duration_s":      round(df["duration"].quantile(0.99), 4),
        "start_date":          start_date,
        "end_date":            end_date,
    }

    print("\n  ── Dataset Summary ──────────────────────────────")
    for k, v in summary.items():
        print(f"    {k:<30}: {v}")
    print("  ─────────────────────────────────────────────────\n")
    return summary


# ── 7. Full pipeline entry point ──────────────────────────────────────────────
def load_pipeline(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full data loading pipeline for the real Azure dataset.

    Returns:
        df          : raw invocation-level DataFrame with cold_start flag
        global_ts   : per-minute global aggregated time series
        summary     : dataset statistics dict
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}\n"
                                f"Place your CSV at this path and re-run.")

    print(f"\n{'='*55}")
    print("  REAL DATA LOADER — Azure Functions CSV")
    print(f"{'='*55}")

    df       = load_sorted_invocations(csv_path)
    df       = add_cold_start_flag(df)
    summary  = dataset_summary(df)
    global_ts = build_global_timeseries(df)

    return df, global_ts, summary


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/azure_functions.csv"
    df, ts, summary = load_pipeline(path)
    print(ts.head(10))