"""
Trace-driven simulator for evaluating cold start mitigation strategies.
Compares:
  1. No pre-warming (all cold starts)
  2. Fixed keep-alive (5-min TTL)
  3. Histogram-based warming (AWS-style)
  4. Our hybrid ML framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SIMULATION_CONFIG


class Container:
    """Simulates a single serverless container lifecycle."""
    def __init__(self, ts_created: int):
        self.created_at = ts_created
        self.last_used = ts_created
        self.is_warm = False  # True once started
        self.startup_time = SIMULATION_CONFIG["container_startup_time_s"]


class Simulator:
    """
    Minute-resolution trace-driven simulator.
    Returns per-minute metrics and aggregate statistics.
    """

    def __init__(self):
        self.cfg = SIMULATION_CONFIG
        self.warm_latency = self.cfg["warm_container_latency_ms"]
        self.cold_latency = self.cfg["cold_container_latency_ms"]
        self.ttl = self.cfg["keep_alive_ttl_minutes"]

    def run_baseline_no_warming(self, invocations: np.ndarray) -> Dict:
        """All invocations experience cold starts."""
        cold_starts = (invocations > 0).sum()
        total = (invocations > 0).sum()
        latencies = np.where(invocations > 0, self.cold_latency, 0)
        return self._build_result(
            "No Warming", invocations, cold_starts, total, latencies,
            containers_warmed=0
        )

    def run_fixed_keepalive(self, invocations: np.ndarray) -> Dict:
        """Keep containers warm for TTL minutes after last use."""
        n = len(invocations)
        cold_starts = 0
        containers_warmed = 0
        last_active = -999
        latencies = np.zeros(n)

        for t in range(n):
            if invocations[t] > 0:
                if (t - last_active) > self.ttl:
                    cold_starts += 1
                    latencies[t] = self.cold_latency
                else:
                    latencies[t] = self.warm_latency
                last_active = t
            # Container is "kept alive" between last_active and last_active+ttl
            is_warm = (t - last_active) <= self.ttl and last_active >= 0
            if is_warm:
                containers_warmed += 1

        total = (invocations > 0).sum()
        return self._build_result(
            "Fixed Keep-Alive (5min TTL)", invocations, cold_starts, total, latencies,
            containers_warmed=containers_warmed
        )

    def run_histogram_warming(self, invocations: np.ndarray,
                               train_size: int = 7 * 24 * 60) -> Dict:
        """
        AWS-style provisioned concurrency: use historical histogram to decide when to warm.
        Warm container at hours/minutes that historically had traffic.
        """
        n = len(invocations)
        cold_starts = 0
        containers_warmed = 0
        latencies = np.zeros(n)

        # Build histogram from training portion
        train = invocations[:train_size]
        minute_of_day_hist = np.zeros(24 * 60)
        for i, v in enumerate(train):
            mod = i % (24 * 60)
            minute_of_day_hist[mod] += v > 0

        # Threshold: warm if >30% of training days had traffic at this minute
        n_training_days = train_size // (24 * 60)
        warm_minutes = set(np.where(minute_of_day_hist / max(n_training_days, 1) > 0.3)[0])

        is_warm = False
        last_active = -999

        for t in range(n):
            mod = t % (24 * 60)
            should_warm = mod in warm_minutes
            if should_warm:
                containers_warmed += 1
                is_warm = True

            if invocations[t] > 0:
                if not is_warm and (t - last_active) > self.ttl:
                    cold_starts += 1
                    latencies[t] = self.cold_latency
                else:
                    latencies[t] = self.warm_latency
                last_active = t

            # TTL keep-alive
            if (t - last_active) > self.ttl:
                is_warm = False

        total = (invocations > 0).sum()
        return self._build_result(
            "Histogram Warming (AWS-style)", invocations, cold_starts, total, latencies,
            containers_warmed=containers_warmed
        )

    def run_ml_framework(self, invocations: np.ndarray,
                          warm_decisions: np.ndarray) -> Dict:
        """
        ML-driven proactive warming.
        warm_decisions: boolean array, True = pre-warm container at time t.
        """
        n = len(invocations)
        cold_starts = 0
        containers_warmed = int(warm_decisions.sum())
        latencies = np.zeros(n)
        is_warm = False
        last_active = -999

        for t in range(n):
            if warm_decisions[t]:
                is_warm = True

            if invocations[t] > 0:
                if not is_warm and (t - last_active) > self.ttl:
                    cold_starts += 1
                    latencies[t] = self.cold_latency
                else:
                    latencies[t] = self.warm_latency
                last_active = t

            # TTL-based expiry
            if (t - last_active) > self.ttl:
                is_warm = False

        total = (invocations > 0).sum()
        return self._build_result(
            "ML Hybrid Framework", invocations, cold_starts, total, latencies,
            containers_warmed=containers_warmed
        )

    def _build_result(self, name: str, invocations: np.ndarray,
                       cold_starts: int, total_invocations: int,
                       latencies: np.ndarray, containers_warmed: int) -> Dict:
        """Compute all evaluation metrics."""
        active_latencies = latencies[latencies > 0]

        # Baseline containers (if no warming, every invocation needs a container)
        baseline_containers = int((invocations > 0).sum())
        resource_overhead = (
            (containers_warmed - baseline_containers) / max(baseline_containers, 1)
        ) if containers_warmed > baseline_containers else 0.0

        efficiency = 0.0
        if containers_warmed > 0:
            utilized = sum(1 for t in range(len(invocations))
                          if invocations[t] > 0)
            efficiency = min(utilized / containers_warmed, 1.0)

        return {
            "strategy": name,
            "total_invocations": int(total_invocations),
            "cold_starts": int(cold_starts),
            "cold_start_rate": cold_starts / max(total_invocations, 1),
            "p50_latency_ms": float(np.percentile(active_latencies, 50)) if len(active_latencies) else 0,
            "p95_latency_ms": float(np.percentile(active_latencies, 95)) if len(active_latencies) else 0,
            "p99_latency_ms": float(np.percentile(active_latencies, 99)) if len(active_latencies) else 0,
            "mean_latency_ms": float(np.mean(active_latencies)) if len(active_latencies) else 0,
            "containers_warmed": containers_warmed,
            "resource_overhead_pct": resource_overhead * 100,
            "resource_efficiency": efficiency,
        }

    @staticmethod
    def compare(results: List[Dict]) -> pd.DataFrame:
        """Format results as comparison table."""
        rows = []
        for r in results:
            rows.append({
                "Strategy": r["strategy"],
                "Cold Start Rate": f"{r['cold_start_rate']*100:.1f}%",
                "P50 Latency (ms)": f"{r['p50_latency_ms']:.0f}",
                "P95 Latency (ms)": f"{r['p95_latency_ms']:.0f}",
                "P99 Latency (ms)": f"{r['p99_latency_ms']:.0f}",
                "Resource Overhead": f"{r['resource_overhead_pct']:.1f}%",
                "Resource Efficiency": f"{r['resource_efficiency']*100:.1f}%",
            })
        return pd.DataFrame(rows)