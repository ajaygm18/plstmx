"""
Configuration settings for PLSTM-TAL stock market prediction model
Based on PMC10963254 research paper
"""

import os

# Data Configuration
STOCK_INDICES = {
    'US': '^GSPC',      # S&P 500
    'UK': '^FTSE',      # FTSE 100
    'China': '000001.SS', # SSE Composite
    'India': '^NSEI'    # NIFTY 50
}

START_DATE = '2005-01-01'  # Full range as per research paper
END_DATE = '2022-12-31'

# Technical Indicators List (40 indicators as specified)
TECHNICAL_INDICATORS = [
    'BBANDS', 'WMA', 'EMA', 'DEMA', 'KAMA', 'MAMA', 'MIDPRICE', 'SAR', 'SMA', 'T3',
    'TEMA', 'TRIMA', 'AD', 'ADOSC', 'OBV', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
    'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX', 'MACD',
    'MFI', 'MINUS_DI', 'MOM', 'PLUS_DI', 'LOG_RETURN', 'PPO', 'ROC', 'RSI',
    'STOCH', 'STOCHRSI', 'ULTOSC', 'WILLR'
]

# Model Configuration
PLSTM_CONFIG = {
    'sequence_length': 60,
    'lstm_units': 128,
    'attention_units': 64,
    'dropout_rate': 0.2,
    'batch_size': 32,
    'epochs': 150,  # More epochs for better training
    'learning_rate': 0.001
}

# Contractive Autoencoder Configuration
CAE_CONFIG = {
    'encoding_dim': 32,
    'lambda_reg': 1e-4,
    'epochs': 50,
    'batch_size': 32
}

# EEMD Configuration
EEMD_CONFIG = {
    'trials': 100,
    'noise_width': 0.2,
    'ext_EMD': None
}

# Bayesian Optimization Configuration
BAYESIAN_OPT_CONFIG = {
    'n_calls': 50,
    'n_initial_points': 10,
    'random_state': 42
}

# Evaluation Configuration
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# File Paths
DATA_DIR = 'data'
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
