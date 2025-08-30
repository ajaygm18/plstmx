# PLSTM-TAL Stock Market Prediction

## Overview

This is a comprehensive stock market prediction system implementing the PLSTM-TAL (Peephole LSTM with Temporal Attention Layer) model based on research paper PMC10963254. The system predicts stock market movements for 4 major indices (S&P 500, FTSE 100, SSE Composite, NIFTY 50) using advanced deep learning techniques combined with technical analysis and signal processing.

The application combines multiple sophisticated approaches: Ensemble Empirical Mode Decomposition (EEMD) for noise reduction, 40 technical indicators for feature engineering, Contractive Autoencoder for dimensionality reduction, and a novel PLSTM-TAL architecture with Bayesian optimization for hyperparameter tuning. It includes benchmark comparisons with CNN, LSTM, SVM, and Random Forest models.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Multi-page interface with navigation sidebar for different functionalities (Overview, Data Processing, Model Training, Evaluation, Results)
- **Interactive Visualizations**: Plotly-based charts and graphs for data exploration and results presentation
- **Real-time Model Training**: Live progress tracking and metrics display during model training

### Backend Architecture
- **Modular Design**: Separate modules for data loading, preprocessing, model implementations, training, and utilities
- **Configuration Management**: Centralized settings for model parameters, technical indicators, and optimization configurations
- **Pipeline Architecture**: Sequential processing from raw data through EEMD filtering, technical indicators calculation, CAE feature extraction, to final model training

### Data Processing Pipeline
- **Multi-Index Data Collection**: Yahoo Finance integration for 4 major stock indices (2005-2022)
- **EEMD Noise Filtering**: Ensemble Empirical Mode Decomposition for signal denoising using sample entropy-based IMF selection
- **Technical Indicators Engine**: 40 technical indicators using TA-Lib library covering overlap studies, momentum indicators, volume indicators, and volatility measures
- **Contractive Autoencoder**: Dimensionality reduction with regularization to extract meaningful features from technical indicators

### Model Architecture
- **PLSTM-TAL Core**: Custom Peephole LSTM with Temporal Attention Layer implementation using TensorFlow/Keras
- **Benchmark Models**: CNN, standard LSTM, SVM, and Random Forest implementations for performance comparison
- **Hyperparameter Optimization**: Bayesian optimization using scikit-optimize for automated parameter tuning
- **Evaluation Framework**: Comprehensive metrics including accuracy, precision, recall, F1-score, AUC-ROC, and Matthews correlation coefficient

### Training and Optimization
- **Bayesian Optimization**: Automated hyperparameter search across LSTM units, attention units, dropout rates, learning rates, batch sizes, and sequence lengths
- **Cross-Validation**: Proper train/validation/test splits with temporal consistency for time series data
- **Model Persistence**: Checkpoint saving and loading capabilities for trained models
- **Performance Tracking**: Training history logging and visualization

## External Dependencies

### Data Sources
- **Yahoo Finance API**: Primary data source for historical stock market data via yfinance library
- **Real-time Data**: Support for live market data integration

### Machine Learning Libraries
- **TensorFlow/Keras**: Core deep learning framework for PLSTM-TAL implementation
- **scikit-learn**: Traditional machine learning models (SVM, Random Forest) and preprocessing utilities
- **scikit-optimize**: Bayesian optimization for hyperparameter tuning
- **PyEMD**: Ensemble Empirical Mode Decomposition implementation

### Technical Analysis
- **TA-Lib**: Technical analysis library for calculating 40 different technical indicators
- **Custom Indicators**: Additional financial metrics and transformations

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
- **StandardScaler**: Feature normalization and scaling

### Visualization and UI
- **Streamlit**: Web application framework for interactive user interface
- **Plotly**: Interactive plotting library for charts and visualizations
- **Plotly Express**: Simplified plotting interface

### Utilities
- **logging**: Comprehensive logging system for debugging and monitoring
- **datetime**: Time-based operations and date handling
- **typing**: Type hints for better code documentation and IDE support