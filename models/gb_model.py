"""
Gradient Boosting classifier/regressor for feature-rich invocation prediction.
Uses XGBoost for handling irregular patterns and contextual signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, FEATURE_CONFIG


class GBInvocationPredictor:
    """
    Multi-horizon gradient boosting predictor.
    Trains separate models for each prediction horizon (1min, 5min, 15min).
    """

    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or FEATURE_CONFIG["prediction_horizons"]
        self.cfg = MODEL_CONFIG["gradient_boosting"]
        self.classifiers: Dict[int, object] = {}
        self.regressors: Dict[int, object] = {}
        self.feature_importances: Dict[int, np.ndarray] = {}
        self._fitted = False

    def _make_clf(self):
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=self.cfg["n_estimators"],
                max_depth=self.cfg["max_depth"],
                learning_rate=self.cfg["learning_rate"],
                subsample=self.cfg["subsample"],
                colsample_bytree=self.cfg["colsample_bytree"],
                min_child_weight=self.cfg["min_child_weight"],
                random_state=self.cfg["random_state"],
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.cfg["n_estimators"],
                max_depth=self.cfg["max_depth"],
                learning_rate=self.cfg["learning_rate"],
                subsample=self.cfg["subsample"],
                random_state=self.cfg["random_state"],
            )

    def _make_reg(self):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=self.cfg["n_estimators"],
                max_depth=self.cfg["max_depth"],
                learning_rate=self.cfg["learning_rate"],
                subsample=self.cfg["subsample"],
                colsample_bytree=self.cfg["colsample_bytree"],
                min_child_weight=self.cfg["min_child_weight"],
                random_state=self.cfg["random_state"],
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.cfg["n_estimators"],
                max_depth=self.cfg["max_depth"],
                learning_rate=self.cfg["learning_rate"],
                subsample=self.cfg["subsample"],
                random_state=self.cfg["random_state"],
            )

    def fit(self, X_train: np.ndarray, y_train: pd.DataFrame,
            feature_names: List[str] = None) -> "GBInvocationPredictor":
        """Train per-horizon classifiers and regressors."""
        for h in self.horizons:
            clf_col = f"label_binary_{h}m"
            reg_col = f"label_count_{h}m"

            if clf_col not in y_train.columns:
                print(f"  Warning: {clf_col} not found, skipping horizon {h}m")
                continue

            y_clf = y_train[clf_col].values
            y_reg = y_train[reg_col].values if reg_col in y_train.columns else y_clf * 5.0

            clf = self._make_clf()
            reg = self._make_reg()

            print(f"  Training GB classifier for {h}m horizon...")
            if len(np.unique(y_clf)) < 2:
                print(f"  Skipping {h}m classifier: only one class in labels")
                continue
            clf.fit(X_train, y_clf)
            print(f"  Training GB regressor for {h}m horizon...")
            reg.fit(X_train, y_reg)

            self.classifiers[h] = clf
            self.regressors[h] = reg

            if hasattr(clf, "feature_importances_"):
                self.feature_importances[h] = clf.feature_importances_

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape (N, n_horizons) probability array."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        probs = []
        for h in self.horizons:
            if h not in self.classifiers:
                probs.append(np.zeros(len(X)))
                continue
            p = self.classifiers[h].predict_proba(X)[:, 1]
            probs.append(p)
        return np.column_stack(probs)

    def predict_counts(self, X: np.ndarray) -> np.ndarray:
        """Return shape (N, n_horizons) count predictions."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        counts = []
        for h in self.horizons:
            if h not in self.regressors:
                counts.append(np.zeros(len(X)))
                continue
            c = self.regressors[h].predict(X).clip(0)
            counts.append(c)
        return np.column_stack(counts)

    def evaluate(self, X: np.ndarray, y_df: pd.DataFrame) -> Dict[str, float]:
        """Compute accuracy, precision, recall, F1 per horizon."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {}
        proba = self.predict_proba(X)
        for i, h in enumerate(self.horizons):
            clf_col = f"label_binary_{h}m"
            if clf_col not in y_df.columns:
                continue
            y_true = y_df[clf_col].values
            y_pred = (proba[:, i] > 0.5).astype(int)
            metrics[f"accuracy_{h}m"] = accuracy_score(y_true, y_pred)
            metrics[f"precision_{h}m"] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f"recall_{h}m"] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f"f1_{h}m"] = f1_score(y_true, y_pred, zero_division=0)
        return metrics

    def top_features(self, horizon: int = 1, top_k: int = 10,
                      feature_names: List[str] = None) -> List[Tuple[str, float]]:
        """Return top-k most important features for a given horizon."""
        if horizon not in self.feature_importances:
            return []
        imp = self.feature_importances[horizon]
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(imp))]
        ranked = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]