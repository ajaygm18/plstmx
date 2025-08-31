# PLSTM-TAL Stock Market Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20%2B-orange)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

Advanced stock market prediction system implementing the research paper **PMC10963254**: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Modes](#training-modes)
- [Results & Output](#results--output)
- [Project Structure](#project-structure)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a **Peephole LSTM with Temporal Attention Layer (PLSTM-TAL)** for high-accuracy stock market price movement prediction. The system combines cutting-edge deep learning techniques with comprehensive financial data analysis.

### Core Components

- 🔍 **Peephole LSTM**: Enhanced LSTM with peephole connections for improved long-term dependencies
- ⏰ **Temporal Attention Layer**: Time-aware attention mechanism for dynamic feature weighting
- 🎯 **Binary Classification**: Predicts price movement direction (up/down) with high accuracy
- 📊 **40 Technical Indicators**: Comprehensive technical analysis features
- 🔄 **EEMD Filtering**: Ensemble Empirical Mode Decomposition for noise reduction
- 🧠 **Contractive Autoencoder**: Advanced feature extraction and dimensionality reduction

### Target Accuracy (from research paper)
- **U.K.**: 96%
- **China**: 88% 
- **U.S.**: 85%
- **India**: 85%

### Stock Indices Supported

- **US**: S&P 500 (^GSPC) - 4531+ historical records
- **UK**: FTSE 100 (^FTSE) - 4545+ historical records
- **China**: SSE Composite (000001.SS) - 4370+ historical records
- **India**: NIFTY 50 (^NSEI) - 3747+ historical records

## Features

### 🚀 **Advanced Machine Learning Pipeline**
- **Data Processing**: Automated OHLCV data collection from 2005-2022
- **Feature Engineering**: 40 technical indicators with validation
- **Signal Processing**: EEMD decomposition with entropy optimization
- **Model Architecture**: State-of-the-art PLSTM-TAL implementation
- **Hyperparameter Optimization**: Bayesian optimization with 500+ calls
- **Benchmarking**: Comparison with CNN, LSTM, SVM, Random Forest

### 🎯 **Multiple Training Modes**
- **Mock Mode**: No dependencies required - instant demo
- **Standard Training**: 300 epochs with standard optimization
- **Extended Training**: 1000 epochs with advanced techniques
- **Maximum Accuracy**: Unlimited resources for 80%+ accuracy target

### 📊 **Comprehensive Evaluation**
- **Accuracy Metrics**: Precision, Recall, F1-Score, AUC-ROC
- **Trading Metrics**: Sharpe ratio, Maximum drawdown, Returns
- **Statistical Analysis**: Confusion matrices, classification reports
- **Visualization**: Interactive plots and performance dashboards

### 🌐 **Web Interface**
- **Streamlit Application**: User-friendly web interface
- **Real-time Predictions**: Live model inference
- **Interactive Visualizations**: Dynamic charts and analysis
- **Model Comparison**: Side-by-side benchmark results

## Requirements

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **Memory**: 8GB RAM minimum (16GB+ recommended for training)
- **Storage**: 5GB free space (10GB+ recommended for full training)
- **GPU**: Optional but recommended (CUDA-compatible for faster training)

### Python Dependencies
```
streamlit>=1.49.1          # Web application framework
pandas>=2.3.2              # Data manipulation and analysis
numpy>=2.3.2               # Numerical computing
tensorflow>=2.20.0         # Deep learning framework
scikit-learn>=1.7.1        # Machine learning utilities
plotly>=6.3.0              # Interactive visualizations
yfinance>=0.2.65           # Financial data API
scikit-optimize>=0.10.2    # Bayesian optimization
seaborn>=0.13.2            # Statistical data visualization
matplotlib>=3.10.6         # Plotting library
EMD-signal>=1.6.4          # Empirical Mode Decomposition
```

## Installation

### Option 1: Quick Start (No Dependencies Required)

Run the application immediately with built-in fallback implementations:

```bash
# Clone the repository
git clone https://github.com/ajaygm18/plstmx.git
cd plstmx

# Run with mock dependencies (no installation required)
python3 run_app.py
```

This option provides:
- ✅ Complete application functionality
- ✅ Mock data and model demonstrations
- ✅ All UI features and visualizations
- ✅ No dependency installation needed

### Option 2: Standard Installation

For full functionality with real data and training:

```bash
# Clone the repository
git clone https://github.com/ajaygm18/plstmx.git
cd plstmx

# Create virtual environment (recommended)
python3 -m venv plstm_env
source plstm_env/bin/activate  # On Windows: plstm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run app.py
```

### Option 3: Advanced Installation with GPU Support

For maximum performance with CUDA GPU acceleration:

```bash
# Clone the repository
git clone https://github.com/ajaygm18/plstmx.git
cd plstmx

# Create virtual environment
python3 -m venv plstm_env
source plstm_env/bin/activate

# Install CUDA-enabled TensorFlow (if you have compatible GPU)
pip install tensorflow-gpu>=2.20.0

# Install other dependencies
pip install -r requirements.txt

# Verify GPU detection
python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Run application
streamlit run app.py
```

## Configuration

### Basic Configuration

The project uses `config/settings.py` for all configuration. Key settings include:

```python
# Stock indices to analyze
STOCK_INDICES = {
    'US': '^GSPC',      # S&P 500
    'UK': '^FTSE',      # FTSE 100
    'China': '000001.SS', # SSE Composite
    'India': '^NSEI'    # NIFTY 50
}

# Data range
START_DATE = '2005-01-01'
END_DATE = '2022-12-31'

# Model configuration
PLSTM_CONFIG = {
    'sequence_length': 60,
    'lstm_units': 256,
    'attention_units': 128,
    'dropout_rate': 0.15,
    'epochs': 1000,
    'batch_size': 32
}
```

### Environment Variables (Optional)

```bash
# Set custom data directory
export PLSTM_DATA_DIR="/path/to/data"

# Set model save directory
export PLSTM_MODELS_DIR="/path/to/models"

# Enable GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set logging level
export PLSTM_LOG_LEVEL=INFO
```

## Usage

### 1. Web Interface (Recommended)

#### Option A: Mock Mode (No Installation)
```bash
python3 run_app.py
```

#### Option B: Full Dependencies
```bash
streamlit run app.py
```

The web interface provides:
- **Interactive Dashboard**: Real-time stock analysis
- **Model Training**: Configure and start training jobs
- **Results Visualization**: Performance metrics and charts
- **Data Exploration**: Technical indicators and market analysis

### 2. Command Line Training

#### Quick Demo Training
```bash
python3 demo_complete_training.py
```

#### Standard Training
```bash
python3 run_extended_training.py
```

#### Maximum Accuracy Training
```bash
python3 run_complete_training.py
```

### 3. Pipeline Testing

#### Test Complete Pipeline
```bash
python3 test_pipeline.py
```

#### Test Enhanced Training
```bash
python3 test_enhanced_training.py
```

#### Test Application Components
```bash
python3 test_app.py
```

## Training Modes

### 1. Mock Mode 🚀
**Purpose**: Instant demonstration without dependencies
```bash
python3 run_app.py
```

**Features**:
- ✅ No installation required
- ✅ Mock data generation
- ✅ Simulated training process
- ✅ Complete UI functionality
- ⏱️ **Execution Time**: Instant

### 2. Standard Training 📊
**Purpose**: Balanced training with good accuracy
```bash
python3 run_extended_training.py
```

**Configuration**:
- 📈 **Epochs**: 300
- 🔍 **Bayesian Calls**: 200
- 🎯 **Target Accuracy**: 70%
- ⏱️ **Execution Time**: 2-4 hours

### 3. Extended Training 🔥
**Purpose**: High-performance training with advanced techniques
```bash
python3 run_complete_training.py
```

**Configuration**:
- 📈 **Epochs**: 1000
- 🔍 **Bayesian Calls**: 500
- 🔄 **EEMD Trials**: 200
- 🎯 **Target Accuracy**: 80%
- 🚀 **Features**: Cosine annealing, warm restarts, ensemble training
- ⏱️ **Execution Time**: 8-24 hours

### 4. Maximum Accuracy Training ⚡
**Purpose**: Unlimited resources for maximum performance
```bash
python3 run_maximum_accuracy_training.py
```

**Configuration**:
- 📈 **Epochs**: Unlimited (with early stopping)
- 🔍 **Bayesian Calls**: 1000+
- 🔄 **EEMD Trials**: 500+
- 🎯 **Target Accuracy**: 85%+ (research paper targets)
- 🚀 **Features**: All advanced techniques enabled
- ⏱️ **Execution Time**: Days (recommended for research/production)

## Results & Output

### Directory Structure
```
results/
├── complete_training/
│   ├── accuracy_results.json          # Main accuracy metrics
│   ├── complete_training_summary.json # Comprehensive results
│   ├── US_training_history.json       # Training progress
│   ├── UK_training_history.json
│   ├── China_training_history.json
│   ├── India_training_history.json
│   ├── US_best_params.json            # Optimal hyperparameters
│   └── *_metrics.json                 # Detailed evaluation metrics

saved_models/
├── plstm_tal/
│   ├── US_plstm_tal_model.keras       # Trained models
│   ├── UK_plstm_tal_model.keras
│   ├── China_plstm_tal_model.keras
│   └── India_plstm_tal_model.keras

logs/
├── tensorboard/                       # TensorBoard logs
├── training_logs/                     # Training progress logs
└── error_logs/                        # Error tracking
```

### Accuracy Results Format

```json
{
  "accuracy_results": {
    "US": {
      "accuracy": 0.8145,
      "precision": 0.7892,
      "recall": 0.8234,
      "f1_score": 0.8059,
      "auc_roc": 0.8567,
      "target_achieved": true,
      "target_accuracy": 0.80
    },
    "UK": {
      "accuracy": 0.8312,
      "target_achieved": true
    }
  },
  "overall_stats": {
    "mean_accuracy": 0.8117,
    "success_rate": 0.75,
    "models_above_target": 3
  }
}
```

### TensorBoard Monitoring

```bash
# Start TensorBoard to monitor training
tensorboard --logdir=logs/tensorboard --port=6006

# View at http://localhost:6006
```

## Project Structure

```
plstmx/
├── 📱 **Application Layer**
│   ├── app.py                          # Main Streamlit application
│   ├── run_app.py                      # Mock dependency runner
│   └── start.sh                        # Quick start script
│
├── ⚙️ **Configuration**
│   └── config/
│       └── settings.py                 # All configuration settings
│
├── 📊 **Data Pipeline**
│   └── data/
│       ├── data_loader.py             # Stock data collection (yfinance)
│       ├── preprocessing.py           # Data preprocessing pipeline
│       └── technical_indicators.py    # 40 technical indicators
│
├── 🧠 **Models**
│   └── models/
│       ├── plstm_tal.py              # Main PLSTM-TAL implementation
│       └── benchmark_models.py        # CNN, LSTM, SVM, Random Forest
│
├── 🎓 **Training Pipeline**
│   ├── training/
│   │   └── trainer.py                 # Training orchestration
│   ├── run_complete_training.py       # Maximum accuracy training
│   ├── run_extended_training.py       # Extended training mode
│   ├── run_maximum_accuracy_training.py # Unlimited resources mode
│   └── demo_complete_training.py      # Quick demonstration
│
├── 🔧 **Utilities**
│   └── utils/
│       ├── evaluation_metrics.py      # Comprehensive evaluation
│       ├── contractive_autoencoder.py # Feature extraction
│       ├── eemd_decomposition.py      # Noise reduction
│       └── bayesian_optimization.py   # Hyperparameter tuning
│
├── 🧪 **Testing**
│   ├── test_app.py                    # Application testing
│   ├── test_pipeline.py               # Pipeline testing
│   ├── test_enhanced_training.py      # Training testing
│   ├── test_comprehensive.py          # Comprehensive testing
│   └── test_final_demo.py             # Final demonstration
│
├── 🎭 **Fallback System**
│   └── mocks.py                       # Mock implementations
│
├── 📋 **Documentation**
│   ├── README.md                      # This file
│   ├── PMC10963254_IMPLEMENTATION.md  # Research paper implementation
│   ├── TESTING_RESULTS.md            # Testing documentation
│   └── training_results_summary.md   # Training results
│
└── 📦 **Dependencies**
    ├── requirements.txt               # Python dependencies
    ├── pyproject.toml                # Project configuration
    └── uv.lock                       # Dependency lock file
```

## Advanced Configuration

### Custom Model Configuration

Edit `config/settings.py` to customize model parameters:

```python
# Enhanced PLSTM Configuration
PLSTM_CONFIG = {
    'sequence_length': 60,      # Input sequence length
    'lstm_units': 256,          # LSTM hidden units
    'attention_units': 128,     # Attention layer units
    'dropout_rate': 0.15,       # Dropout for regularization
    'learning_rate': 0.0005,    # Learning rate
    'batch_size': 32,           # Training batch size
    'epochs': 1000,             # Maximum training epochs
    'patience': 150,            # Early stopping patience
}

# Extended Training Configuration
EXTENDED_TRAINING_CONFIG = {
    'target_accuracy': 0.80,     # Target accuracy (80%)
    'cosine_annealing': True,    # Cosine annealing LR schedule
    'warm_restarts': True,       # Warm restart training
    'ensemble_training': True,   # Ensemble techniques
    'data_augmentation': True,   # Data augmentation
}

# Bayesian Optimization Configuration
BAYESIAN_OPT_CONFIG = {
    'n_calls': 500,             # Number of optimization calls
    'n_initial_points': 50,     # Initial random points
    'acq_func': 'EI',           # Acquisition function
    'n_jobs': -1,               # Parallel jobs (-1 = all cores)
}
```

### Custom Stock Indices

```python
# Add custom stock indices
CUSTOM_INDICES = {
    'Japan': '^N225',          # Nikkei 225
    'Germany': '^GDAXI',       # DAX
    'France': '^FCHI',         # CAC 40
    'Australia': '^AXJO',      # ASX 200
}

# Update in config/settings.py
STOCK_INDICES.update(CUSTOM_INDICES)
```

### Performance Optimization

```python
# GPU Configuration
import tensorflow as tf

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision training (advanced)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Install dependencies
pip install -r requirements.txt

# Alternative: Use mock mode
python3 run_app.py
```

#### 2. Memory Issues
```bash
# Problem: Out of memory during training
# Solutions:
# 1. Reduce batch size in config/settings.py
PLSTM_CONFIG['batch_size'] = 16  # Reduce from 32

# 2. Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 3. Use CPU-only training
export CUDA_VISIBLE_DEVICES=""
```

#### 3. Training Too Slow
```bash
# Solutions:
# 1. Use GPU acceleration
pip install tensorflow-gpu

# 2. Reduce training parameters
EXTENDED_TRAINING_CONFIG['epochs'] = 300  # Reduce from 1000
BAYESIAN_OPT_CONFIG['n_calls'] = 200      # Reduce from 500

# 3. Use standard training mode
python3 run_extended_training.py  # Instead of complete training
```

#### 4. Data Download Issues
```bash
# Problem: yfinance connection errors
# Solutions:
# 1. Check internet connection
# 2. Use mock mode for offline usage
python3 run_app.py

# 3. Manually set proxy (if needed)
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

#### 5. TensorFlow Issues
```bash
# Problem: TensorFlow warnings/errors
# Solutions:
# 1. Update TensorFlow
pip install --upgrade tensorflow>=2.20.0

# 2. Install specific version
pip install tensorflow==2.20.0

# 3. Use CPU-only version
pip install tensorflow-cpu
```

### Performance Tips

1. **Hardware Recommendations**:
   - **CPU**: Intel i7/AMD Ryzen 7+ with 8+ cores
   - **RAM**: 16GB+ for full training, 8GB minimum
   - **GPU**: NVIDIA RTX 3060+ with 8GB+ VRAM (optional)
   - **Storage**: SSD recommended for faster data loading

2. **Optimization Settings**:
   ```python
   # In config/settings.py
   PLSTM_CONFIG['batch_size'] = 64    # Increase if you have more RAM
   BAYESIAN_OPT_CONFIG['n_jobs'] = 8  # Set to your CPU core count
   ```

3. **Monitoring Resources**:
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1
   
   # Monitor CPU/RAM usage
   htop
   
   # Monitor training progress
   tensorboard --logdir=logs/tensorboard
   ```

### Getting Help

1. **Check Logs**:
   ```bash
   # View training logs
   tail -f logs/training_logs/latest.log
   
   # View error logs
   tail -f logs/error_logs/latest.log
   ```

2. **Debug Mode**:
   ```bash
   # Run with debug logging
   export PLSTM_LOG_LEVEL=DEBUG
   python3 run_complete_training.py
   ```

3. **Test Components**:
   ```bash
   # Test individual components
   python3 test_pipeline.py          # Test data pipeline
   python3 test_enhanced_training.py # Test training pipeline
   python3 test_app.py               # Test application
   ```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/ajaygm18/plstmx.git
cd plstmx

# Create development environment
python3 -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python3 -m pytest

# Format code
black .

# Lint code
flake8 .
```

### Research Paper Reference

This implementation is based on:
**"Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"** (PMC10963254)

### License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.

---

**📧 Support**: For questions or issues, please check the [troubleshooting section](#troubleshooting) or open an issue on GitHub.

**🚀 Quick Start**: `python3 run_app.py` - Get started in seconds with no installation required!