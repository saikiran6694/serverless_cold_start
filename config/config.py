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
    "sequence_length": 60,                # LSTM lookback in minutes
    "temporal_features": [
        "hour", "day_of_week", "minute",
        "is_business_hour", "is_weekend",
        "week_of_year", "month"
    ],
}

# Model Hyperparameters
MODEL_CONFIG = {
    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50,
        "early_stopping_patience": 10,
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
        "lstm_weight": 0.5,
        "gb_weight": 0.5,
    }
}

# Adaptive Threshold Controller
THRESHOLD_CONFIG = {
    "initial_threshold": 0.5,
    "min_threshold": 0.2,
    "max_threshold": 0.9,
    "learning_rate": 0.01,
    "accuracy_window": 100,             # recent predictions to evaluate
    "target_cold_start_rate": 0.10,     # 10%
    "target_resource_efficiency": 0.70, # 70%
    "adjustment_step": 0.05,
}

# Simulation
SIMULATION_CONFIG = {
    "container_startup_time_s": 2.0,     # cold start penalty
    "warm_container_latency_ms": 5,
    "cold_container_latency_ms": 2500,
    "keep_alive_ttl_minutes": 5,         # baseline
    "resource_cost_per_container_hour": 1.0,
}

# Targets (from proposal)
TARGETS = {
    "cold_start_reduction_pct": 70,      # 60-80% range
    "p95_latency_improvement_pct": 85,   # 80-90% range
    "resource_overhead_pct": 25,         # below 30%
}