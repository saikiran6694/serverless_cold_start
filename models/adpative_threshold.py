"""
Adaptive Threshold Controller.
Dynamically adjusts warming threshold based on prediction accuracy and resource utilization.
Targets: 10% cold start rate, 70% resource efficiency.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import THRESHOLD_CONFIG


class AdaptiveThresholdController:
    """
    PID-inspired adaptive threshold controller.

    Logic:
    - If cold start rate > target → lower threshold (be more aggressive in warming)
    - If resource waste > target → raise threshold (be more conservative)
    - Balance using recent accuracy window
    """

    def __init__(self):
        cfg = THRESHOLD_CONFIG
        self.threshold = cfg["initial_threshold"]
        self.min_threshold = cfg["min_threshold"]
        self.max_threshold = cfg["max_threshold"]
        self.lr = cfg["learning_rate"]
        self.step = cfg["adjustment_step"]
        self.window = cfg["accuracy_window"]
        self.target_cold_start = cfg["target_cold_start_rate"]
        self.target_efficiency = cfg["target_resource_efficiency"]

        # Rolling buffers
        self.recent_predictions = deque(maxlen=self.window)   # (prob, actual)
        self.recent_warm_decisions = deque(maxlen=self.window) # (warmed, used)
        self.threshold_history = [self.threshold]
        self.cold_start_rate_history = []
        self.efficiency_history = []

    def record_prediction(self, prob: float, actual: bool):
        """Record a prediction and its outcome."""
        self.recent_predictions.append((prob, actual))

    def record_warm_decision(self, was_warmed: bool, was_invoked: bool):
        """Record whether a container was warmed and if it was actually needed."""
        self.recent_warm_decisions.append((was_warmed, was_invoked))

    def compute_cold_start_rate(self) -> float:
        """Fraction of invocations that experienced a cold start (not warmed but invoked)."""
        if not self.recent_warm_decisions:
            return 1.0
        cold_starts = sum(1 for w, i in self.recent_warm_decisions if not w and i)
        invocations = sum(1 for w, i in self.recent_warm_decisions if i)
        if invocations == 0:
            return 0.0
        return cold_starts / invocations

    def compute_resource_efficiency(self) -> float:
        """Fraction of warmed containers that were actually used."""
        if not self.recent_warm_decisions:
            return 0.5
        warmed = sum(1 for w, i in self.recent_warm_decisions if w)
        used = sum(1 for w, i in self.recent_warm_decisions if w and i)
        if warmed == 0:
            return 1.0
        return used / warmed

    def compute_prediction_accuracy(self) -> float:
        """Binary accuracy of recent predictions at current threshold."""
        if not self.recent_predictions:
            return 0.5
        correct = sum(
            1 for prob, actual in self.recent_predictions
            if (prob >= self.threshold) == actual
        )
        return correct / len(self.recent_predictions)

    def update(self) -> Tuple[float, Dict[str, float]]:
        """
        Update threshold based on current cold start rate and efficiency.
        Returns (new_threshold, metrics_dict).
        """
        cold_start_rate = self.compute_cold_start_rate()
        efficiency = self.compute_resource_efficiency()
        accuracy = self.compute_prediction_accuracy()

        self.cold_start_rate_history.append(cold_start_rate)
        self.efficiency_history.append(efficiency)

        # Compute deltas from targets
        cold_start_delta = cold_start_rate - self.target_cold_start      # positive → too many cold starts
        efficiency_delta = self.target_efficiency - efficiency             # positive → too many wasted warms

        # Adjust threshold
        if cold_start_delta > 0.05:
            # Too many cold starts → lower threshold (warm more aggressively)
            adjustment = -self.step * min(cold_start_delta / 0.2, 1.0)
        elif efficiency_delta > 0.1:
            # Too many wasted containers → raise threshold (warm less)
            adjustment = +self.step * min(efficiency_delta / 0.3, 1.0)
        else:
            # Near target: fine-grained adjustment
            net = cold_start_delta * 0.6 - efficiency_delta * 0.4
            adjustment = -self.lr * net

        new_threshold = np.clip(self.threshold + adjustment, self.min_threshold, self.max_threshold)
        self.threshold = new_threshold
        self.threshold_history.append(self.threshold)

        metrics = {
            "threshold": self.threshold,
            "cold_start_rate": cold_start_rate,
            "resource_efficiency": efficiency,
            "prediction_accuracy": accuracy,
            "adjustment": adjustment,
        }
        return self.threshold, metrics

    def should_warm(self, prob: float) -> bool:
        """Decision: should we pre-warm a container given this probability?"""
        return prob >= self.threshold

    def should_warm_multi_horizon(self, probs: np.ndarray,
                                   horizon_weights: np.ndarray = None) -> bool:
        """
        Multi-horizon warm decision.
        Uses confidence-weighted ensemble of horizon probabilities.
        """
        if horizon_weights is None:
            # Higher weight for shorter horizons (more confident)
            horizon_weights = np.array([0.5, 0.3, 0.2])
        weighted_prob = np.dot(probs, horizon_weights)
        return weighted_prob >= self.threshold

    def get_state(self) -> Dict:
        return {
            "threshold": self.threshold,
            "cold_start_rate": self.cold_start_rate_history[-1] if self.cold_start_rate_history else None,
            "efficiency": self.efficiency_history[-1] if self.efficiency_history else None,
            "n_predictions_seen": len(self.recent_predictions),
        }