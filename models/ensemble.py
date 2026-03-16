"""
Hybrid Ensemble: combines LSTM and Gradient Boosting predictions
with confidence-weighted fusion.
"""

import numpy as np
from typing import Optional, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, FEATURE_CONFIG
from models.adpative_threshold import AdaptiveThresholdController


class HybridEnsemble:
    """
    Fuses LSTM and GB predictions with learnable weights.
    Supports dynamic weight adjustment based on recent accuracy.
    """

    def __init__(self, lstm_weight: float = None, gb_weight: float = None):
        cfg = MODEL_CONFIG["ensemble"]
        self.lstm_weight = lstm_weight or cfg["lstm_weight"]
        self.gb_weight = gb_weight or cfg["gb_weight"]
        assert abs(self.lstm_weight + self.gb_weight - 1.0) < 1e-6, "Weights must sum to 1"

        self.controller = AdaptiveThresholdController()
        self.horizons = FEATURE_CONFIG["prediction_horizons"]
        self.horizon_weights = np.array([0.5, 0.3, 0.2])  # 1min > 5min > 15min

    def fuse(self, lstm_probs: Optional[np.ndarray],
             gb_probs: Optional[np.ndarray]) -> np.ndarray:
        """
        Combine LSTM and GB predictions.
        Falls back gracefully if one model is unavailable.
        """
        if lstm_probs is None and gb_probs is None:
            raise ValueError("At least one model must provide predictions.")
        if lstm_probs is None:
            return gb_probs
        if gb_probs is None:
            return lstm_probs

        # Ensure same shape
        min_len = min(len(lstm_probs), len(gb_probs))
        lstm_probs = lstm_probs[:min_len]
        gb_probs = gb_probs[:min_len]

        return self.lstm_weight * lstm_probs + self.gb_weight * gb_probs

    def predict_warm_decisions(self, ensemble_probs: np.ndarray) -> np.ndarray:
        """
        For each time step, decide whether to pre-warm using multi-horizon logic.
        Returns boolean array of warm decisions.
        """
        decisions = np.array([
            self.controller.should_warm_multi_horizon(
                probs, self.horizon_weights
            )
            for probs in ensemble_probs
        ])
        return decisions

    def update_controller(self, warm_decisions: np.ndarray,
                           actual_invocations: np.ndarray) -> Dict:
        """Feed outcomes back to adaptive threshold controller."""
        for warmed, actual in zip(warm_decisions, actual_invocations):
            self.controller.record_warm_decision(bool(warmed), bool(actual > 0))
        _, metrics = self.controller.update()
        return metrics

    def record_predictions_for_accuracy(self, probs_1m: np.ndarray,
                                          actuals: np.ndarray):
        """Store 1-minute ahead predictions for threshold accuracy tracking."""
        for p, a in zip(probs_1m, actuals):
            self.controller.record_prediction(float(p), bool(a > 0))

    def adjust_weights(self, lstm_recent_acc: float, gb_recent_acc: float):
        """Dynamically rebalance ensemble weights based on recent per-model accuracy."""
        total = lstm_recent_acc + gb_recent_acc
        if total < 1e-6:
            return
        self.lstm_weight = lstm_recent_acc / total
        self.gb_weight = gb_recent_acc / total

    @property
    def threshold(self) -> float:
        return self.controller.threshold

    def get_summary(self) -> Dict:
        return {
            "lstm_weight": self.lstm_weight,
            "gb_weight": self.gb_weight,
            "current_threshold": self.controller.threshold,
            "controller_state": self.controller.get_state(),
        }