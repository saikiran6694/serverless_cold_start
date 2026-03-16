"""
EDA visualizations specific to the real Azure Functions dataset.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_cold_start_eda(df: pd.DataFrame, output_path: str, sample_n: int = 200_000):
    """
    Cold start EDA: rates by hour, day, function — from raw invocation table.
    """
    # Sample for speed if huge
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    df = df.copy()
    df["hour"]       = df["start_dt"].dt.hour
    df["day_of_week"]= df["start_dt"].dt.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cold Start EDA — Real Azure Functions Data",
                 fontsize=13, fontweight="bold")

    # By hour
    by_hour = df.groupby("hour")["cold_start"].mean() * 100
    axes[0].bar(by_hour.index, by_hour.values, color="#3498DB", edgecolor="white")
    axes[0].set_title("Cold Start Rate by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Cold Start Rate (%)")
    axes[0].axhline(by_hour.mean(), color="red", linestyle="--", linewidth=1,
                     label=f"Mean: {by_hour.mean():.1f}%")
    axes[0].legend()

    # By day of week
    by_dow = df.groupby("day_of_week")["cold_start"].mean() * 100
    colors = ["#27AE60" if i < 5 else "#E74C3C" for i in range(7)]
    axes[1].bar([day_names[i] for i in by_dow.index], by_dow.values,
                 color=[colors[i] for i in by_dow.index], edgecolor="white")
    axes[1].set_title("Cold Start Rate by Day of Week")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Cold Start Rate (%)")

    # Duration distribution (warm vs cold)
    warm = df[df["cold_start"] == 0]["duration"].clip(upper=30)
    cold = df[df["cold_start"] == 1]["duration"].clip(upper=30)
    axes[2].hist(warm, bins=60, alpha=0.6, color="#27AE60", label="Warm Start", density=True)
    axes[2].hist(cold, bins=60, alpha=0.6, color="#E74C3C", label="Cold Start", density=True)
    axes[2].set_title("Duration Distribution\n(Warm vs Cold Starts)")
    axes[2].set_xlabel("Duration (s, clipped at 30s)")
    axes[2].set_ylabel("Density")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_duration_distribution(df: pd.DataFrame, output_path: str,
                                sample_n: int = 500_000):
    """Full duration distribution with percentiles."""
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    durations = df["duration"].clip(upper=df["duration"].quantile(0.99))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Function Duration Distribution — Real Data",
                 fontsize=13, fontweight="bold")

    axes[0].hist(durations, bins=100, color="#3498DB", edgecolor="none", alpha=0.8)
    axes[0].set_title("Duration Histogram (capped at P99)")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count")
    for pct, color in [(50, "green"), (95, "orange"), (99, "red")]:
        val = df["duration"].quantile(pct / 100)
        axes[0].axvline(val, color=color, linestyle="--", linewidth=1.2,
                         label=f"P{pct}: {val:.2f}s")
    axes[0].legend()

    # Log-scale
    axes[1].hist(df["duration"].clip(lower=1e-4), bins=100,
                  color="#9B59B6", edgecolor="none", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_title("Duration Histogram (log scale)")
    axes[1].set_xlabel("Duration (s, log scale)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cold_start_rate_over_time(global_ts: pd.DataFrame, output_path: str):
    """Cold start rate and volume over the full trace window."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Cold Start Rate Over Time — Real Trace",
                 fontsize=13, fontweight="bold")

    ts = global_ts["timestamp"]
    axes[0].fill_between(ts, global_ts["total_invocations"],
                          alpha=0.5, color="#3498DB", label="Total Invocations")
    axes[0].fill_between(ts, global_ts["cold_starts"],
                          alpha=0.7, color="#E74C3C", label="Cold Starts")
    axes[0].set_ylabel("Invocations / min")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Invocation Volume with Cold Starts Highlighted")

    # Rolling cold start rate
    roll_rate = global_ts["cold_start_rate"].rolling(30, min_periods=1).mean() * 100
    axes[1].plot(ts, roll_rate, color="#E74C3C", linewidth=1.2, label="30-min rolling avg")
    axes[1].axhline(roll_rate.mean(), color="black", linestyle="--",
                     linewidth=1, label=f"Overall mean: {roll_rate.mean():.1f}%")
    axes[1].set_ylabel("Cold Start Rate (%)")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Cold Start Rate (30-min rolling average)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_function_heatmap(func_ts: pd.DataFrame, output_path: str):
    """Heatmap of per-function invocation counts over time."""
    pivot = func_ts.pivot_table(
        index="func", columns="timestamp",
        values="invocations", aggfunc="sum", fill_value=0
    )
    # Truncate func IDs for display
    pivot.index = [str(f)[:12] + "…" for f in pivot.index]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Per-Function Invocation Heatmap (Top Functions)",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, label="Invocations/min")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")