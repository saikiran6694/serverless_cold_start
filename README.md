# Proactive Cold Start Mitigation in Serverless Environments

A machine learning pipeline for predicting and mitigating cold starts in serverless cloud computing environments using the Huawei Public Cloud Trace 2025 dataset.

## Overview

This project implements a comprehensive 31-day machine learning pipeline using PyTorch and scikit-learn to:
- **Predict cold starts** in serverless environments
- **Forecast** container cold start rates at multiple time horizons (1-min, 5-min, 15-min)
- **Optimize** resource allocation through an adaptive threshold controller
- **Ensemble** multiple models for improved prediction accuracy

**Dataset:** Huawei Public Cloud Trace  Region 1 Cold Start Traces2025 

## Architecture

The system consists of five key components:

1. **Bidirectional LSTM ( Captures sequential temporal patterns in cold start eventsPyTorch)** 
2. **RandomForest (scikit- Extracts insights from tabular engineered featureslearn)** 
3. **Confidence-weighted  Combines predictions weighted by validation AUC scoresEnsemble** 
4. **Adaptive Threshold  Dynamically adjusts decision thresholds based on FP/FN trade-offsController** 
5. **Multi-horizon  Generates forecasts at 1-minute, 5-minute, and 15-minute intervalsPrediction** 

## Data Structure

The pipeline processes 31 days of continuous trace data:

- **Training Set:** Days 18 (19 days)0
- **Validation Set:** Days 24 (6 days)19
- **Test Set:** Days 30 (6 days, held-out evaluation)25

Each day's data is organized as per-minute records with:
- `total`: Total invocations
- `cold`: Cold start count
- `cold_rate`: Cold start ratio
- Temporal features (hour, day_of_week, business hours)
- Rolling window statistics (5-min, 15-min, 60-min)
- Lagged features (1, 5, 15, 30-minute lags)

## Installation

### Prerequisites

-  3.14Python 
- pip or uv package manager

### Setup

1. **Install dependencies:**

```bash
# if uv
uv add -r requirements.txt

# or else
pip install -r requirements.txt
```

For GPU acceleration with CUDA support:

```bash
# if uv
uv add torch --index-url https://download.pytorch.org/whl/cu121

# or else
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

2. **Download and prepare data:**

The dataset consists of 31 CSV files (day_0.csv through day_30.csv) representing one month of Huawei cloud trace data.

**Download the dataset:**

Click the link below to download the R1.zip file containing all trace data:
([Download R1 Dataset](https://drive.google.com/file/d/1mMQtfZNtg-EPmGmGYuOzC5KPbZoXRd8e/view?usp=sharing))


**Extract the dataset:**

```bash
unzip R1.zip
```

This will create a directory with CSV files named `day_0.csv`, `day_1.csv`, ..., `day_30.csv`.

## Usage

Run the full pipeline:

```bash
# if uv
uv run python main.py --dir /path/to/extracted/data/

# or else
python main.py --dir /path/to/extracted/data/
```

**Arguments:**
- `--dir`: Path to directory containing extracted CSV files (day_0.csv through day_30.csv)

**Example:**
```bash
# if uv
uv run python main.py --dir ./extracted_data/

# or else
python main.py --dir ./extracted_data/
```

### Output

The pipeline generates:

- **Trained models** in the `models/` directory
- **Evaluation metrics** and visualizations in the `results/` directory
- **Predictions** for validation and test sets
- **Performance reports** with ROC-AUC, F1, precision, and recall scores

## Features & Configuration

### Hyperparameters

Edit `main.py` to customize:

- `LSTM_EPOCHS`: Number of training epochs (default: 120)
- `BATCH_SIZE`: Batch size for training (default: 64)
- `PATIENCE`: Early stopping patience (default: 20)
- `COLD_THRESHOLD`: Cold start classification threshold (default: 0.25)
- `SEQUENCE_LEN`: LSTM input sequence length (default: 30)

### Primary Horizon

The system focuses on 5-minute predictions by default (`PRIMARY_HORIZON = 'target_5min'`). Modify to use 1-minute or 15-minute horizons as needed.

### Available Features

**Temporal Features:**
- Hour of day, day of week, business hours indicator

**Aggregation Features:**
- 5, 15, and 60-minute rolling means, standard deviations, and cold rate

**Lagged Features:**
- 1, 5, 15, and 30-minute lagged totals and cold counts

**Derived Features:**
- Trend indicators, burst ratios, cold acceleration metrics

## Performance Metrics

The pipeline evaluates using:

- **ROC-AUC**: Area under the receiver operating characteristic curve
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)

Results are generated for each prediction horizon and the ensemble model.

## Technical Stack

- **PyTorch:** Deep learning and LSTM implementation
- **scikit-learn:** Random Forest, metrics, preprocessing
- **pandas:** Data manipulation and feature engineering
- **NumPy:** Numerical computations
- **Matplotlib:** Visualization and results plotting
- **XGBoost & LightGBM:** Optional gradient boosting models
- **TensorFlow/Keras:** Conditional support for Python 3.3.1310

## Device Support

The system automatically detects and uses:
- **CUDA GPU** if available (recommended for faster training)
- **CPU** as fallback

Check device usage in training logs.

## File Structure

```
 main.py                 # Main pipeline script
 requirements.txt        # Python dependencies
 pyproject.toml         # Project configuration
 README.md              # This file
 R1.zip                 # Dataset archive (after download)
 models/                # Trained model artifacts
 results/               # Output predictions and visualizations
 uv.lock               # Dependency lock file
```

## Troubleshooting

**Issue: "day_0.csv not found"**
- Ensure the dataset is downloaded and extracted
- Verify the `--dir` path points to the extracted CSV files directory
- Check that files are named day_0.csv through day_30.csv

**Issue: Download fails**
- Check your internet connection
- Try using curl instead of wget: `curl -O <URL>`
- Verify the download link is active

**Issue: CUDA out of memory**
- Reduce `BATCH_SIZE` in `main.py`
- Use CPU by not installing CUDA PyTorch

**Issue: Slow training**
- Verify GPU is being used: check logs for CUDA device
- Increase `BATCH_SIZE` if GPU memory allows

## Citation

If you use this project, please cite:

```
Proactive Cold Start Mitigation in Serverless Environments
Dataset: Huawei Public Cloud Trace  Region 12025 
```

## License

See project repository for licensing information.

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/
- scikit-learn: https://scikit-learn.org/
- Huawei Cloud Traces: [[Github](https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md)]
