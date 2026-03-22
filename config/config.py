"""
Configuration for Proactive Cold Start Mitigation System
"""

# Data
DATA_CONFIG = {
    "trace_duration_days": 14,
    "aggregation_interval_minutes": 1,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
}

# Feature Engineering
FEATURE_CONFIG = {
    "rolling_windows": [5, 15, 60],       # minutes
    "prediction_horizons": [1, 5, 15],    # minutes ahead
    "sequence_length": 30,                # LSTM lookback (reduced from 60 → faster CPU)
    "temporal_features": [
        "hour", "day_of_week", "minute",
        "is_business_hour", "is_weekend",
        "week_of_year", "month"
    ],
}

# Model Hyperparameters
MODEL_CONFIG = {
    "lstm": {
        "hidden_size": 64,        # reduced from 128 → ~4x faster, negligible accuracy loss
        "num_layers": 1,          # reduced from 2  → removes inter-layer dropout overhead
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 256,        # increased from 64 → fewer gradient steps per epoch
        "epochs": 20,             # reduced from 50 → early stopping handles quality
        "early_stopping_patience": 5,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
    },
    "ensemble": {
        "lstm_weight": 0.4,   # LSTM: temporal patterns
        "gb_weight": 0.6,     # GB:   feature-rich, dominant weight
    }
}

# Adaptive Threshold Controller
THRESHOLD_CONFIG = {
    "initial_threshold": 0.5,
    "min_threshold": 0.2,
    "max_threshold": 0.9,
    "learning_rate": 0.01,
    "accuracy_window": 100,
    "target_cold_start_rate": 0.10,
    "target_resource_efficiency": 0.70,
    "adjustment_step": 0.05,
}

# Simulation
SIMULATION_CONFIG = {
    "container_startup_time_s": 2.0,
    "warm_container_latency_ms": 5,
    "cold_container_latency_ms": 2500,
    "keep_alive_ttl_minutes": 5,
    "resource_cost_per_container_hour": 1.0,
}

# Targets (from proposal)
TARGETS = {
    "cold_start_reduction_pct": 70,
    "p95_latency_improvement_pct": 85,
    "resource_overhead_pct": 25,
}