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
from typing import Dict, List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SIMULATION_CONFIG


class Simulator:
    """
    Minute-resolution trace-driven simulator.

    Key concepts
    ------------
    - warm_slots   : number of minutes where a container was kept warm
                     proactively (i.e. a cost the strategy pays).
    - used_slots   : warm_slots that coincided with an actual invocation
                     (the benefit received).
    - resource_overhead : (warm_slots - actual_invocations) / actual_invocations
                          — how many *extra* container-minutes were paid for
                          beyond bare-minimum reactive spinning.
    - resource_efficiency : used_slots / warm_slots
                            — fraction of proactive warm time that was utilised.

    The "No Warming" baseline never pays for proactive warm time, so its
    overhead = 0% and efficiency = 0% (no warm slots were allocated at all).
    """

    def __init__(self):
        self.cfg = SIMULATION_CONFIG
        self.warm_latency = self.cfg["warm_container_latency_ms"]
        self.cold_latency = self.cfg["cold_container_latency_ms"]
        self.ttl = self.cfg["keep_alive_ttl_minutes"]

    # ------------------------------------------------------------------ #
    #  Strategy 1 — No warming (pure reactive)                            #
    # ------------------------------------------------------------------ #
    def run_baseline_no_warming(self, invocations: np.ndarray) -> Dict:
        """Every invocation experiences a cold start — zero proactive cost."""
        total = int((invocations > 0).sum())
        latencies = np.where(invocations > 0, self.cold_latency, 0)
        return self._build_result(
            "No Warming", invocations,
            cold_starts=total, total_invocations=total,
            latencies=latencies, warm_slots=0
        )

    # ------------------------------------------------------------------ #
    #  Strategy 2 — Fixed keep-alive (TTL after each invocation)          #
    # ------------------------------------------------------------------ #
    def run_fixed_keepalive(self, invocations: np.ndarray) -> Dict:
        """
        After each invocation keep the container alive for TTL minutes.
        warm_slots counts every minute the container is kept warm beyond
        the invocation minute itself (idle holding cost).
        """
        n = len(invocations)
        cold_starts = 0
        latencies = np.zeros(n)
        last_active = -999
        # Count container-minutes that are proactively held (idle warm slots)
        idle_warm_slots = 0

        for t in range(n):
            gap = t - last_active
            is_warm = (gap <= self.ttl) and (last_active >= 0)

            if invocations[t] > 0:
                if not is_warm:
                    cold_starts += 1
                    latencies[t] = self.cold_latency
                else:
                    latencies[t] = self.warm_latency
                last_active = t
            else:
                # Minute where no invocation arrived but container is held warm
                if is_warm:
                    idle_warm_slots += 1

        total = int((invocations > 0).sum())
        return self._build_result(
            "Fixed Keep-Alive (5min TTL)", invocations,
            cold_starts=cold_starts, total_invocations=total,
            latencies=latencies, warm_slots=idle_warm_slots
        )

    # ------------------------------------------------------------------ #
    #  Strategy 3 — Histogram / AWS-style provisioned concurrency         #
    # ------------------------------------------------------------------ #
    def run_histogram_warming(self, invocations: np.ndarray,
                               train_size: int = 7 * 24 * 60) -> Dict:
        """
        Build a minute-of-day histogram from the training window.
        Pre-warm during minutes that historically had traffic > 30% of days.
        warm_slots = proactive pre-warm minutes that had no invocation.
        """
        n = len(invocations)
        cold_starts = 0
        latencies = np.zeros(n)

        # Build histogram from training portion
        train = invocations[:train_size]
        minute_of_day_hist = np.zeros(24 * 60)
        for i, v in enumerate(train):
            minute_of_day_hist[i % (24 * 60)] += int(v > 0)

        n_training_days = max(train_size // (24 * 60), 1)
        warm_set = set(
            np.where(minute_of_day_hist / n_training_days > 0.3)[0]
        )

        is_warm = False
        last_active = -999
        idle_warm_slots = 0

        for t in range(n):
            # Pre-warm decision from histogram
            if t % (24 * 60) in warm_set:
                is_warm = True

            if invocations[t] > 0:
                if not is_warm and (t - last_active) > self.ttl:
                    cold_starts += 1
                    latencies[t] = self.cold_latency
                else:
                    latencies[t] = self.warm_latency
                last_active = t
            else:
                if is_warm:
                    idle_warm_slots += 1

            # TTL expiry after last use
            if (t - last_active) > self.ttl and t % (24 * 60) not in warm_set:
                is_warm = False

        total = int((invocations > 0).sum())
        return self._build_result(
            "Histogram Warming (AWS-style)", invocations,
            cold_starts=cold_starts, total_invocations=total,
            latencies=latencies, warm_slots=idle_warm_slots
        )

    # ------------------------------------------------------------------ #
    #  Strategy 4 — ML hybrid framework                                   #
    # ------------------------------------------------------------------ #
    def run_ml_framework(self, invocations: np.ndarray,
                          warm_decisions: np.ndarray) -> Dict:
        """
        ML-driven proactive warming.
        warm_decisions[t] = True  →  pre-warm container at minute t.

        A warm decision activates the container for a short ML_TTL window
        (shorter than the fixed keep-alive TTL) to reflect that the model
        predicts only a near-term invocation — not indefinite warmth.
        idle_warm_slots counts pre-warm minutes where no invocation arrived.
        """
        ML_TTL = max(self.ttl - 2, 2)   # ML warms more precisely: shorter window
        n = len(invocations)
        cold_starts = 0
        latencies = np.zeros(n)
        last_warm_decision = -999
        last_active = -999
        idle_warm_slots = 0

        for t in range(n):
            if warm_decisions[t]:
                last_warm_decision = t

            # Container is warm if ML recently decided to warm it
            ml_warm = (t - last_warm_decision) <= ML_TTL and last_warm_decision >= 0
            # Also warm from TTL keep-alive after an actual invocation
            ttl_warm = (t - last_active) <= self.ttl and last_active >= 0
            is_warm = ml_warm or ttl_warm

            if invocations[t] > 0:
                if not is_warm:
                    cold_starts += 1
                    latencies[t] = self.cold_latency
                else:
                    latencies[t] = self.warm_latency
                last_active = t
            else:
                if ml_warm and not ttl_warm:
                    # Proactively warmed by ML but no invocation arrived
                    idle_warm_slots += 1

        total = int((invocations > 0).sum())
        return self._build_result(
            "ML Hybrid Framework", invocations,
            cold_starts=cold_starts, total_invocations=total,
            latencies=latencies, warm_slots=idle_warm_slots
        )

    # ------------------------------------------------------------------ #
    #  Internal metrics builder                                           #
    # ------------------------------------------------------------------ #
    def _build_result(self, name: str, invocations: np.ndarray,
                       cold_starts: int, total_invocations: int,
                       latencies: np.ndarray, warm_slots: int) -> Dict:
        """
        Compute evaluation metrics.

        resource_overhead_pct
            Extra container-minutes paid per actual invocation (%).
            = (idle warm slots / actual invocations) * 100
            Captures wasted proactive warming — lower is better.

        resource_efficiency
            Fraction of proactive warm minutes that coincided with an
            actual invocation.
            = actual_invocations / (actual_invocations + idle_warm_slots)
            Higher is better; 0% for No Warming (no warm slots allocated).
        """
        active_latencies = latencies[latencies > 0]
        actual_inv = int((invocations > 0).sum())

        # Overhead: idle warm slots as a fraction of actual invocations
        resource_overhead = warm_slots / max(actual_inv, 1)

        # Efficiency: invocations served warm out of all warm-allocated minutes
        total_warm_minutes = actual_inv + warm_slots  # minutes where container was warm
        if total_warm_minutes > 0:
            efficiency = actual_inv / total_warm_minutes
        else:
            efficiency = 0.0  # No Warming: no warm minutes at all

        return {
            "strategy": name,
            "total_invocations": int(total_invocations),
            "cold_starts": int(cold_starts),
            "cold_start_rate": cold_starts / max(total_invocations, 1),
            "p50_latency_ms": float(np.percentile(active_latencies, 50)) if len(active_latencies) else 0,
            "p95_latency_ms": float(np.percentile(active_latencies, 95)) if len(active_latencies) else 0,
            "p99_latency_ms": float(np.percentile(active_latencies, 99)) if len(active_latencies) else 0,
            "mean_latency_ms": float(np.mean(active_latencies)) if len(active_latencies) else 0,
            "warm_slots": warm_slots,
            "resource_overhead_pct": resource_overhead * 100,
            "resource_efficiency": efficiency,
        }

    # ------------------------------------------------------------------ #
    #  Comparison table                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compare(results: List[Dict]) -> pd.DataFrame:
        """Format results as a human-readable comparison table."""
        rows = []
        for r in results:
            rows.append({
                "Strategy":           r["strategy"],
                "Cold Start Rate":    f"{r['cold_start_rate']*100:.1f}%",
                "P50 Latency (ms)":   f"{r['p50_latency_ms']:.0f}",
                "P95 Latency (ms)":   f"{r['p95_latency_ms']:.0f}",
                "P99 Latency (ms)":   f"{r['p99_latency_ms']:.0f}",
                "Resource Overhead":  f"{r['resource_overhead_pct']:.1f}%",
                "Resource Efficiency":f"{r['resource_efficiency']*100:.1f}%",
            })
        return pd.DataFrame(rows)