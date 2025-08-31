# PLSTM-TAL Stock Market Prediction

Implementation of the research paper **PMC10963254**: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"

## Overview

This project implements a Peephole LSTM with Temporal Attention Layer (PLSTM-TAL) for stock market price movement prediction. The model combines:

- 🔍 **Peephole LSTM**: Enhanced LSTM with peephole connections
- ⏰ **Temporal Attention Layer**: Time-aware attention mechanism  
- 🎯 **Binary Classification**: Predicts price movement direction (up/down)

### Target Accuracy (from research paper)
- **U.K.**: 96%
- **China**: 88% 
- **U.S.**: 85%
- **India**: 85%

## Features

- **Data Processing Pipeline**: Historical OHLCV data collection, 40 technical indicators, EEMD filtering
- **Feature Extraction**: Contractive Autoencoder for dimensionality reduction
- **Model Training**: PLSTM-TAL architecture with Bayesian hyperparameter optimization
- **Evaluation**: Comprehensive metrics and benchmark comparison
- **Web Interface**: Streamlit application for interactive use

## Stock Indices Supported

- **US**: S&P 500 (^GSPC)
- **UK**: FTSE 100 (^FTSE)  
- **China**: SSE Composite (000001.SS)
- **India**: NIFTY 50 (^NSEI)

## Quick Start

### Option 1: Run with Mock Dependencies (No Installation Required)

```bash
python3 run_app.py
```

This runs the complete application with fallback implementations, demonstrating all functionality without requiring dependency installation.

### Option 2: Run with Full Dependencies 

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run app.py
```

## Project Structure

```
plstmx/
├── app.py                          # Main Streamlit application
├── config/
│   └── settings.py                 # Configuration settings
├── data/
│   ├── data_loader.py             # Stock data collection
│   ├── preprocessing.py           # Data preprocessing pipeline
│   └── technical_indicators.py    # 40 technical indicators
├── models/
│   ├── plstm_tal.py              # PLSTM-TAL model implementation
│   └── benchmark_models.py        # Benchmark models (CNN, LSTM, SVM, RF)
├── training/
│   └── trainer.py                 # Training pipeline
├── utils/
│   ├── evaluation_metrics.py      # Comprehensive evaluation metrics
│   ├── contractive_autoencoder.py # Feature extraction
│   ├── eemd_decomposition.py      # Noise reduction
│   └── bayesian_optimization.py   # Hyperparameter tuning
├── mocks.py                       # Fallback implementations
├── run_app.py                     # Mock dependency runner
└── test_app.py                    # Import testing
```

## Technical Implementation

### Data Processing Pipeline
1. **Data Collection**: Historical OHLCV data from yfinance
2. **Technical Indicators**: 40 indicators (RSI, MACD, Bollinger Bands, etc.)
3. **EEMD Filtering**: Ensemble Empirical Mode Decomposition for noise reduction
4. **Feature Extraction**: Contractive Autoencoder for dimensionality reduction
5. **Sequence Creation**: Time series sequences for LSTM training

### Model Architecture
- **Input**: 60-day sequences of processed features
- **Peephole LSTM**: Two-layer LSTM with peephole connections
- **Temporal Attention**: Attention mechanism for temporal pattern focus
- **Classification**: Dense layers for binary classification (up/down movement)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, PR-AUC, Matthews Correlation Coefficient
- Confusion Matrix, Classification Report
- Trading-specific metrics (Sharpe ratio, Maximum drawdown)

## Dependencies

Core dependencies:
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `tensorflow` - Deep learning framework
- `scikit-learn` - Machine learning utilities
- `plotly` - Interactive visualizations
- `yfinance` - Stock data collection

Optional dependencies (with fallbacks):
- `scikit-optimize` - Bayesian optimization
- `PyEMD` - Empirical Mode Decomposition
- `seaborn` - Statistical visualizations

## Fallback System

The project includes comprehensive fallback implementations that allow it to run without installing any dependencies:

- **TensorFlow Fallback**: Simple models with mock training/prediction
- **PyEMD Fallback**: Basic noise filtering using random noise
- **Scikit-optimize Fallback**: Random search instead of Bayesian optimization
- **Data Fallback**: Mock stock data generation

## Research Paper Reference

This implementation is based on:
**"Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"** (PMC10963254)

## License

This project is for educational and research purposes.