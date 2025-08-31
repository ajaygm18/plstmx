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

# Model Configuration - Enhanced for extended training and 70%+ accuracy
PLSTM_CONFIG = {
    'sequence_length': 60,
    'lstm_units': 128,
    'attention_units': 64,
    'dropout_rate': 0.2,
    'batch_size': 32,
    'epochs': 300,  # Increased from 150 for extended training
    'learning_rate': 0.001,
    'patience': 50,  # Early stopping patience for extended training
    'min_delta': 0.0001,  # Minimum improvement for early stopping
    'restore_best_weights': True,  # Restore best weights after training
    'verbose': 1,  # Enable progress output
    'validation_split': 0.2,  # Validation split for training
    'monitor': 'val_accuracy',  # Monitor validation accuracy
    'mode': 'max',  # Maximize validation accuracy
    'save_best_only': True,  # Save only the best model
    'save_weights_only': False,  # Save entire model
    'load_weights_on_restart': True  # Load best weights if training is restarted
}

# Contractive Autoencoder Configuration
CAE_CONFIG = {
    'encoding_dim': 32,
    'lambda_reg': 1e-4,
    'epochs': 50,
    'batch_size': 32
}

# EEMD Configuration - Enhanced for better noise reduction as per PMC10963254
EEMD_CONFIG = {
    'trials': 200,  # Increased from 100 for better ensemble averaging
    'noise_width': 0.15,  # Reduced from 0.2 for more precise decomposition
    'ext_EMD': None,
    'n_imfs': None,  # Let algorithm determine optimal number of IMFs
    'ensemble_mean': True,  # Use ensemble mean for better decomposition
    'extrema_detection': 'parabola',  # Enhanced extrema detection
    'range_thr': 0.001,  # Range threshold for decomposition
    'total_power_thr': 0.05  # Total power threshold
}

# Bayesian Optimization Configuration - Enhanced for PMC10963254 requirements
BAYESIAN_OPT_CONFIG = {
    'n_calls': 200,  # Increased from 50 for more thorough optimization
    'n_initial_points': 20,  # Increased initial points for better exploration
    'random_state': 42,
    'acq_func': 'EI',  # Expected Improvement acquisition function
    'n_jobs': -1,  # Use all available cores
    'verbose': True,  # Enable verbose output for long training sessions
    'timeout': None  # Disable timeouts for extended training
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

# Extended Training Configuration for PMC10963254 compliance
EXTENDED_TRAINING_CONFIG = {
    'epochs': 300,  # Extended training epochs for achieving 70%+ accuracy
    'max_training_time': None,  # No time limit - disable timeouts
    'checkpoint_interval': 50,  # Save checkpoint every 50 epochs
    'backup_interval': 100,  # Backup model every 100 epochs
    'enable_tensorboard': True,  # Enable TensorBoard logging
    'log_dir': 'logs',  # TensorBoard log directory
    'reduce_lr_on_plateau': True,  # Reduce learning rate on plateau
    'lr_reduction_factor': 0.5,  # Learning rate reduction factor
    'lr_patience': 30,  # Patience for learning rate reduction
    'min_lr': 1e-7,  # Minimum learning rate
    'target_accuracy': 0.70,  # Target accuracy of 70%+ as required
    'accuracy_patience': 100,  # Patience to reach target accuracy
    'memory_growth': True,  # Enable memory growth for TensorFlow
    'mixed_precision': False,  # Disable mixed precision for stability
    'gradient_clipping': True,  # Enable gradient clipping
    'max_gradient_norm': 1.0  # Maximum gradient norm
}

# Timeout Configuration - Disable all timeouts for extended training
TIMEOUT_CONFIG = {
    'training_timeout': None,  # No training timeout
    'optimization_timeout': None,  # No optimization timeout
    'evaluation_timeout': None,  # No evaluation timeout
    'data_loading_timeout': None,  # No data loading timeout
    'model_saving_timeout': None,  # No model saving timeout
    'tensorflow_timeout': None,  # No TensorFlow operation timeout
    'streamlit_server_timeout': 0,  # Disable Streamlit server timeout
    'max_execution_time': None  # No maximum execution time
}
