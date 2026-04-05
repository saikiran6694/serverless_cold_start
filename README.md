# Proactive Cold Start Mitigation in Serverless Environments

A comprehensive machine learning pipeline for predicting and mitigating cold starts in serverless cloud computing environments using the Huawei Public Cloud Trace 2025 dataset.

## 📋 Overview

This project implements a sophisticated 31-day machine learning pipeline using PyTorch and scikit-learn to:
- **Predict cold starts** in serverless environments with high accuracy
- **Forecast** container cold start rates at multiple time horizons (1-min, 5-min, 15-min)
- **Optimize** resource allocation through an adaptive threshold controller
- **Ensemble** multiple models for improved prediction accuracy and robustness

**Dataset:** Huawei Public Cloud Trace 2025 – Region 1 Cold Start Traces

## 🏗️ Architecture

The system is built on a hybrid architecture with five key components:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Bidirectional LSTM** | PyTorch | Captures sequential temporal patterns in cold start events |
| **RandomForest Classifier** | scikit-learn | Extracts insights from engineered tabular features |
| **Confidence-Weighted Ensemble** | Custom | Combines predictions weighted by validation AUC scores |
| **Adaptive Threshold Controller** | Custom | Dynamically adjusts decision thresholds based on FP/FN trade-offs |
| **Multi-Horizon Predictor** | PyTorch + scikit-learn | Generates forecasts at 1-minute, 5-minute, and 15-minute intervals |

## 📊 Data Structure

The pipeline processes 31 days of continuous Huawei cloud trace data with explicit train/validation/test splits:

| Dataset | Days | Records | Purpose |
|---------|------|---------|---------|
| **Training** | 0–18 | 19 days | Model training and parameter optimization |
| **Validation** | 19–24 | 6 days | Hyperparameter tuning and model selection |
| **Test** | 25–30 | 6 days | Final held-out evaluation and performance assessment |

### Per-Minute Records Include:
- `total`: Total function invocations
- `cold`: Cold start count
- `cold_rate`: Cold start ratio (cold / total)
- **Temporal features:** hour of day, day of week, business hours indicator
- **Aggregation features:** 5-min, 15-min, 60-min rolling means, standard deviations, and cold rates
- **Lagged features:** 1-min, 5-min, 15-min, 30-min lagged totals and cold counts
- **Derived features:** trend indicators, burst ratios, cold acceleration metrics

## 🚀 Installation

### Prerequisites

- **Python:** 3.14 or higher
- **Package Manager:** pip or [uv](https://github.com/astral-sh/uv) (recommended for speed)

### Setup Steps

1. **Clone the repository and install dependencies:**

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

2. **(Optional) Install GPU support for CUDA 12.1:**

```bash
# Using uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or using pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3. **Download and extract the dataset:**

The dataset consists of 31 CSV files (day_0.csv through day_30.csv) representing one month of Huawei cloud traces.

**Download:** [R1 Dataset](https://drive.google.com/file/d/1mMQtfZNtg-EPmGmGYuOzC5KPbZoXRd8e/view?usp=sharing)

**Extract:**
```bash
unzip R1.zip
```

This creates a directory with CSV files named `day_0.csv`, `day_1.csv`, ..., `day_30.csv`.

## ▶️ Usage

### Running the Full Pipeline

```bash
# Using uv
uv run python main.py --dir /path/to/data/

# Using python
python main.py --dir /path/to/data/
```

### Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dir` | Path to directory containing extracted CSV files | `./R1/` or `/data/traces/` |

### Example

```bash
uv run python main.py --dir ./R1/
```

### Pipeline Outputs

The pipeline generates comprehensive results in the following directories:

- **`models/`** – Trained model artifacts (BiLSTM, RandomForest, ensemble weights)
- **`results/`** – Predictions, evaluation metrics, and performance visualizations
- **Console output** – Training logs, validation metrics, and final performance reports

**Key metrics reported:**
- ROC-AUC scores for each horizon and ensemble
- F1, Precision, and Recall scores
- ROC curves and threshold analysis plots
- Prediction accuracy on held-out test set

## ⚙️ Configuration & Customization

### Hyperparameters

Edit `main.py` to modify:

```python
LSTM_EPOCHS = 120          # Number of training epochs
BATCH_SIZE = 64            # Batch size for LSTM training
PATIENCE = 20              # Early stopping patience
COLD_THRESHOLD = 0.25      # Cold start classification threshold
SEQUENCE_LEN = 30          # LSTM input sequence length (in minutes)
PRIMARY_HORIZON = 'target_5min'  # Primary prediction horizon
```

### Feature Configuration

**Available prediction horizons:**
- `target_1min` – 1-minute ahead forecast
- `target_5min` – 5-minute ahead forecast (default)
- `target_15min` – 15-minute ahead forecast

Modify `PRIMARY_HORIZON` in `main.py` to switch horizons.

### Feature Engineering Details

The pipeline automatically generates:

| Feature Category | Examples |
|------------------|----------|
| **Temporal** | hour_of_day, day_of_week, is_business_hours |
| **Aggregation** | rolling_mean_5m, rolling_std_15m, cold_rate_60m |
| **Lagged** | lag_1m_total, lag_5m_cold, lag_30m_total |
| **Derived** | trend, burst_ratio, cold_acceleration |

## 📈 Performance Metrics

The pipeline evaluates models using standard ML metrics:

| Metric | Definition | Use Case |
|--------|-----------|----------|
| **ROC-AUC** | Area under receiver operating characteristic curve | Overall model discrimination ability |
| **F1 Score** | Harmonic mean of precision and recall | Balanced performance metric |
| **Precision** | TP / (TP + FP) | Minimize false alarms |
| **Recall** | TP / (TP + FN) | Minimize missed cold starts |

Results are generated for each prediction horizon individually and for the ensemble model.

## 🛠️ Technical Stack

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Deep Learning | PyTorch | ≥2.2.0 | LSTM model and training |
| ML & Metrics | scikit-learn | ≥1.3.0 | RandomForest, evaluation metrics |
| Data Processing | pandas | ≥2.0.0 | DataFrames and feature engineering |
| Numerical Computing | NumPy | ≥1.24.0 | Array operations |
| Visualization | Matplotlib | ≥3.7.0 | Plots and reports |
| Gradient Boosting | XGBoost | ≥1.7.0 | Optional ensemble component |
| Gradient Boosting | LightGBM | ≥4.0.0 | Optional ensemble component |
| Deep Learning | TensorFlow/Keras | ≥2.13.0 | Optional alternative to PyTorch |

## 💻 Device Support

The system automatically detects and uses the optimal compute device:

- **CUDA GPU** (if available) – Recommended for faster training and inference
- **CPU** (fallback) – Works on any machine, slower but functional

GPU usage is logged during training. Check logs for `CUDA device` messages.

## 📁 Project Structure

```
serverless_cold_start/
├── main.py                          # Main pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata and config
├── README.md                        # This file
├── uv.lock                          # Dependency lock file (for uv)
├── config/
│   ├── config.py                    # Configuration constants
│   └── __init__.py
├── data_loader/
│   ├── loader.py                    # Dataset loading and parsing
│   └── __init__.py
├── feature_engineer/
│   ├── engineer.py                  # Feature extraction and engineering
│   └── __init__.py
├── models/
│   ├── bilstm_model.py             # BiLSTM PyTorch model definition
│   ├── adaptive_threshold_controller.py  # Threshold adaptation logic
│   ├── simulator.py                 # Simulation and evaluation
│   └── __init__.py
├── visualization/
│   ├── plots.py                     # Plotting and visualization functions
│   └── __init__.py
├── generated_models/                # Output directory for trained models
├── results/                         # Output directory for evaluation results
├── R1.zip                           # Dataset archive (after download)
└── R1/                              # Extracted dataset (day_0.csv through day_30.csv)
```

## 🔧 Troubleshooting

### "day_0.csv not found" Error
**Solution:**
- Ensure the R1.zip dataset is downloaded and extracted: `unzip R1.zip`
- Verify the `--dir` path points to the correct extracted directory containing CSV files
- Confirm files are named exactly `day_0.csv`, `day_1.csv`, ..., `day_30.csv`

### Download Fails
**Solution:**
- Check your internet connection
- Verify the Google Drive link is accessible and not quota-limited
- Try alternative download methods (browser, curl, wget)
- Check available disk space (requires ~500MB for dataset)

### CUDA Out of Memory Error
**Solution:**
- Reduce `BATCH_SIZE` in `main.py` (try 32 or 16)
- Switch to CPU by using non-CUDA PyTorch: `pip install torch`
- Use a GPU with higher VRAM or split the data into smaller chunks

### Slow Training / Low GPU Utilization
**Solution:**
- Verify CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Check training logs for CUDA device usage
- Increase `BATCH_SIZE` if GPU memory allows (up to 256 or 512)
- Use a machine with better GPU hardware (A100, H100, RTX 4090)

### Import Errors (TensorFlow, XGBoost, etc.)
**Solution:**
- Reinstall all dependencies: `pip install -r requirements.txt --force-reinstall`
- Verify Python version is 3.14+: `python --version`
- Check for version conflicts in `pyproject.toml` (TensorFlow compatibility constraints)

## 📚 References

- **PyTorch Documentation:** https://pytorch.org/docs/stable/
- **scikit-learn Guide:** https://scikit-learn.org/stable/
- **Huawei Cloud Traces:** https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md
- **LSTM/BiLSTM Theory:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Ensemble Methods:** https://scikit-learn.org/stable/modules/ensemble.html

## ❓ Support & Questions

For issues, questions, or feature requests:
1. Check the Troubleshooting section above
2. Review existing GitHub Issues
3. Create a new issue with:
   - Error message and full traceback
   - Python version and OS
   - Steps to reproduce the problem
   - Hardware specs (GPU/CPU, RAM)
