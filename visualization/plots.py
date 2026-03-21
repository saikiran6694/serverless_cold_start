"""
Visualization for EDA, model performance, and simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional
import os


COLORS = {
    "No Warming": "#E74C3C",
    "Fixed Keep-Alive (5min TTL)": "#F39C12",
    "Histogram Warming (AWS-style)": "#3498DB",
    "ML Hybrid Framework": "#27AE60",
}
PALETTE = ["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6", "#F39C12", "#1ABC9C"]


def plot_invocation_patterns(global_df: pd.DataFrame, output_path: str):
    """EDA: invocation time series with daily patterns."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Azure Functions: Invocation Patterns (14-day trace)", fontsize=14, fontweight="bold")

    ts = pd.to_datetime(global_df["timestamp"])
    inv = global_df["total_invocations"].values

    # Full series
    axes[0].fill_between(range(len(inv)), inv, alpha=0.6, color="#3498DB")
    axes[0].set_title("Per-minute Invocation Count (Full Trace)")
    axes[0].set_xlabel("Minutes")
    axes[0].set_ylabel("Invocations")

    # Single day — use however many minutes are actually available (up to 1440)
    day_data = inv[:min(24 * 60, len(inv))]
    n_day = len(day_data)
    axes[1].plot(range(n_day), day_data, color="#E74C3C", linewidth=1.2)
    axes[1].set_title(f"Single Day Pattern (first {n_day} minutes)")
    axes[1].set_xlabel("Minute of Day")
    axes[1].set_ylabel("Invocations")
    if n_day > 9 * 60:
        axes[1].axvspan(9 * 60, min(17 * 60, n_day), alpha=0.1, color="green", label="Business Hours")
        axes[1].legend()

    # Hourly heatmap (7 days)
    hourly = np.zeros((7, 24))
    for i in range(min(7 * 24 * 60, len(inv))):
        day = i // (24 * 60)
        hour = (i % (24 * 60)) // 60
        if day < 7:
            hourly[day, hour] += inv[i]
    im = axes[2].imshow(hourly, aspect="auto", cmap="YlOrRd")
    axes[2].set_title("Hourly Invocation Heatmap (7 days)")
    axes[2].set_xlabel("Hour of Day")
    axes[2].set_ylabel("Day of Week")
    axes[2].set_yticks(range(7))
    axes[2].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.colorbar(im, ax=axes[2], label="Invocations")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cold_start_comparison(results: List[Dict], output_path: str):
    """Bar chart comparing cold start rates across strategies."""
    strategies = [r["strategy"] for r in results]
    cold_rates = [r["cold_start_rate"] * 100 for r in results]
    colors = [COLORS.get(s, "#7F8C8D") for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Strategy Performance Comparison", fontsize=14, fontweight="bold")

    # Cold start rate
    bars = axes[0].bar(range(len(strategies)), cold_rates, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].axhline(10, color="green", linestyle="--", linewidth=1.5, label="Target: 10%")
    axes[0].set_title("Cold Start Rate (%)")
    axes[0].set_ylabel("Cold Start Rate (%)")
    axes[0].set_xticks(range(len(strategies)))
    axes[0].set_xticklabels([s.split("(")[0].strip() for s in strategies], rotation=20, ha="right", fontsize=9)
    axes[0].legend()
    for bar, val in zip(bars, cold_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Latency comparison
    p50 = [r["p50_latency_ms"] for r in results]
    p95 = [r["p95_latency_ms"] for r in results]
    x = np.arange(len(strategies))
    width = 0.35
    axes[1].bar(x - width/2, p50, width, label="P50", color=[c + "CC" for c in [
        "#E74C3C", "#F39C12", "#3498DB", "#27AE60"]][:len(strategies)], edgecolor="white")
    axes[1].bar(x + width/2, p95, width, label="P95", color=colors, edgecolor="white")
    axes[1].set_title("Latency Percentiles (ms)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.split("(")[0].strip() for s in strategies], rotation=20, ha="right", fontsize=9)
    axes[1].legend()

    # Resource efficiency
    efficiencies = [r["resource_efficiency"] * 100 for r in results]
    bars2 = axes[2].bar(range(len(strategies)), efficiencies, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].axhline(70, color="blue", linestyle="--", linewidth=1.5, label="Target: 70%")
    axes[2].set_title("Resource Efficiency (%)")
    axes[2].set_ylabel("Efficiency (%)")
    axes[2].set_xticks(range(len(strategies)))
    axes[2].set_xticklabels([s.split("(")[0].strip() for s in strategies], rotation=20, ha="right", fontsize=9)
    axes[2].legend()
    for bar, val in zip(bars2, efficiencies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_training_history(history: Dict, model_name: str, output_path: str):
    """Plot train/val loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} Training History", fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#3498DB")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#E74C3C")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="#27AE60")
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].axhline(0.7, color="orange", linestyle="--", label="Baseline (70%)")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_adaptive_threshold(controller, output_path: str):
    """Plot threshold evolution alongside cold start rate and efficiency."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Adaptive Threshold Controller Evolution", fontsize=13, fontweight="bold")

    t_hist = controller.threshold_history
    cs_hist = controller.cold_start_rate_history
    eff_hist = controller.efficiency_history

    axes[0].plot(t_hist, color="#9B59B6", linewidth=1.5)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Initial (0.5)")
    axes[0].set_ylabel("Threshold")
    axes[0].set_title("Threshold Adaptation")
    axes[0].legend()

    if cs_hist:
        axes[1].plot(cs_hist, color="#E74C3C", linewidth=1.5)
        axes[1].axhline(0.10, color="green", linestyle="--", label="Target 10%")
        axes[1].set_ylabel("Cold Start Rate")
        axes[1].legend()

    if eff_hist:
        axes[2].plot(eff_hist, color="#27AE60", linewidth=1.5)
        axes[2].axhline(0.70, color="blue", linestyle="--", label="Target 70%")
        axes[2].set_ylabel("Resource Efficiency")
        axes[2].set_xlabel("Update Step")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                              title: str, output_path: str, top_k: int = 15):
    """Horizontal bar chart of feature importances."""
    idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    ax.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_dashboard(results: List[Dict], metrics: Dict, output_path: str):
    """Final summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Proactive Cold Start Mitigation — Summary Dashboard", fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    strategies = [r["strategy"] for r in results]
    colors = [COLORS.get(s, "#7F8C8D") for s in strategies]
    short_names = [s.split("(")[0].strip().replace(" ", "\n") for s in strategies]

    # Cold start rate
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(strategies)), [r["cold_start_rate"]*100 for r in results],
                    color=colors, edgecolor="white")
    ax1.axhline(10, color="green", linestyle="--", linewidth=1.5)
    ax1.set_title("Cold Start Rate (%)", fontweight="bold")
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(short_names, fontsize=8)
    for b, r in zip(bars, results):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                  f"{r['cold_start_rate']*100:.1f}%", ha="center", fontsize=8)

    # P95 latency
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(len(strategies)), [r["p95_latency_ms"] for r in results],
                    color=colors, edgecolor="white")
    ax2.set_title("P95 Latency (ms)", fontweight="bold")
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(short_names, fontsize=8)
    for b, r in zip(bars, results):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                  f"{r['p95_latency_ms']:.0f}", ha="center", fontsize=8)

    # Resource efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(range(len(strategies)), [r["resource_efficiency"]*100 for r in results],
                    color=colors, edgecolor="white")
    ax3.axhline(70, color="blue", linestyle="--", linewidth=1.5)
    ax3.set_title("Resource Efficiency (%)", fontweight="bold")
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels(short_names, fontsize=8)
    for b, r in zip(bars, results):
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                  f"{r['resource_efficiency']*100:.1f}%", ha="center", fontsize=8)

    # ML-specific metrics table
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.axis("off")
    ml_result = next((r for r in results if "ML" in r["strategy"]), None)
    baseline = next((r for r in results if r["strategy"] == "No Warming"), None)
    if ml_result and baseline:
        cs_reduction = (1 - ml_result["cold_start_rate"] / max(baseline["cold_start_rate"], 1e-9)) * 100
        p95_improvement = (1 - ml_result["p95_latency_ms"] / max(baseline["p95_latency_ms"], 1e-9)) * 100
        table_data = [
            ["Metric", "Achieved", "Target", "Status"],
            ["Cold Start Reduction", f"{cs_reduction:.1f}%", "60-80%",
             "✓" if 60 <= cs_reduction <= 100 else "~"],
            ["P95 Latency Improvement", f"{p95_improvement:.1f}%", "80-90%",
             "✓" if 80 <= p95_improvement <= 100 else "~"],
            ["Resource Overhead", f"{ml_result['resource_overhead_pct']:.1f}%", "<30%",
             "✓" if ml_result['resource_overhead_pct'] < 30 else "~"],
            ["Resource Efficiency", f"{ml_result['resource_efficiency']*100:.1f}%", ">70%",
             "✓" if ml_result['resource_efficiency'] >= 0.7 else "~"],
        ]
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax4.set_title("Project Goals vs Achieved Metrics", fontweight="bold", pad=10)

    # GB accuracy bar
    ax5 = fig.add_subplot(gs[1, 2])
    if "gb_metrics" in metrics:
        gm = metrics["gb_metrics"]
        horizons = [1, 5, 15]
        accs = [gm.get(f"accuracy_{h}m", 0) for h in horizons]
        bars = ax5.bar([f"{h}min" for h in horizons], [a*100 for a in accs],
                        color=["#27AE60", "#3498DB", "#9B59B6"])
        ax5.axhline(70, color="orange", linestyle="--", linewidth=1.5, label="Baseline 70%")
        ax5.set_title("GB Prediction Accuracy\nper Horizon", fontweight="bold")
        ax5.set_ylabel("Accuracy (%)")
        ax5.legend(fontsize=8)
        for b, a in zip(bars, accs):
            ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                      f"{a*100:.1f}%", ha="center", fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")