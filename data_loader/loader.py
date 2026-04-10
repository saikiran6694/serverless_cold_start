import pandas as pd
import os

def load_day(path: str) -> pd.DataFrame:
    """
    Load one CSV → per-minute (day, minute, total, cold, cold_rate).
    Dataset: Huawei Public Cloud Trace 2025 – Region 1.
    """
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