# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Fixed research-grade splits — explicitly defined for reproducibility
# Train: days 0–18  (19 days)
# Val  : days 19–24  (6 days)
# Test : days 25–30  (6 days, held-out evaluation)
TRAIN_DAYS = list(range(0, 19))    # days 0–18
VAL_DAYS   = list(range(19, 25))   # days 19–24
TEST_DAYS  = list(range(25, 31))   # days 25–30

COLD_THRESHOLD = 0.25
SEQUENCE_LEN   = 30
LSTM_EPOCHS    = 120      # increased: more epochs since we have 31 days of data
BATCH_SIZE     = 64
PATIENCE       = 20       # increased: allow more epochs without improvement before stopping

HORIZONS = {
    'target_1min':  1,
    'target_5min':  5,
    'target_15min': 15,
}
PRIMARY_HORIZON = 'target_5min'

DEVICE = "cpu"

TABULAR_FEATURES = [
    'hour', 'is_business', 'day_of_week',
    'roll_mean_5',  'roll_std_5',  'roll_cold_5',
    'roll_mean_15', 'roll_std_15', 'roll_cold_15',
    'roll_mean_60', 'roll_std_60', 'roll_cold_60',
    'lag_total_1',  'lag_cold_1',
    'lag_total_5',  'lag_cold_5',
    'lag_total_15', 'lag_cold_15',
    'lag_total_30', 'lag_cold_30',
    'trend_5', 'trend_15',
    'burst_ratio',
    'cold_acceleration',
]

SEQUENCE_FEATURES = [
    'total_norm', 'cold_rate',
    'roll_mean_5',  'roll_cold_5',
    'roll_mean_15', 'roll_cold_15',
    'trend_5',
]