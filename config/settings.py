"""
Configuration settings for PLSTM-TAL stock market prediction model
Based on PMC10963254 research paper
"""

import os

# Data Configuration - Modified to use only US index data as requested
STOCK_INDICES = {
    'US': '^GSPC',      # S&P 500 - Only US data as specified in requirements
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

# Model Configuration - Optimized for MAXIMUM ACCURACY with unlimited resources
PLSTM_CONFIG = {
    'sequence_length': 60,
    'lstm_units': 256,  # Increased from 128 for maximum capacity
    'attention_units': 128,  # Increased from 64 for enhanced attention
    'dropout_rate': 0.15,  # Reduced from 0.2 for maximum learning capacity
    'batch_size': 32,
    'epochs': 1000,  # Increased from 300 for maximum training
    'learning_rate': 0.0005,  # Reduced for more stable convergence
    'patience': 150,  # Increased patience for maximum training
    'min_delta': 0.00001,  # More sensitive improvement detection
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

# EEMD Configuration - Optimized for maximum accuracy with 200 trials
EEMD_CONFIG = {
    'trials': 200,  # Optimized for balance between quality and performance
    'noise_width': 0.10,  # Further reduced from 0.15 for highest precision decomposition
    'ext_EMD': None,
    'n_imfs': None,  # Let algorithm determine optimal number of IMFs
    'ensemble_mean': True,  # Use ensemble mean for better decomposition
    'extrema_detection': 'parabola',  # Enhanced extrema detection
    'range_thr': 0.0005,  # Reduced threshold for maximum precision
    'total_power_thr': 0.02  # Reduced for maximum signal capture
}

# Bayesian Optimization Configuration - Maximized for exhaustive hyperparameter optimization
BAYESIAN_OPT_CONFIG = {
    'n_calls': 500,  # Increased from 200 for exhaustive optimization
    'n_initial_points': 50,  # Increased initial points for maximum exploration
    'random_state': 42,
    'acq_func': 'EI',  # Expected Improvement acquisition function
    'n_jobs': -1,  # Use all available cores
    'verbose': True,  # Enable verbose output for long training sessions
    'timeout': None  # Disable timeouts for unlimited optimization
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

# Extended Training Configuration - Optimized for MAXIMUM ACCURACY with unlimited resources
EXTENDED_TRAINING_CONFIG = {
    'epochs': 1000,  # Increased from 300 for maximum training duration
    'max_training_time': None,  # No time limit - unlimited execution
    'checkpoint_interval': 25,  # More frequent checkpoints every 25 epochs
    'backup_interval': 50,  # More frequent backups every 50 epochs
    'enable_tensorboard': True,  # Enable TensorBoard logging
    'log_dir': 'logs',  # TensorBoard log directory
    'reduce_lr_on_plateau': True,  # Reduce learning rate on plateau
    'lr_reduction_factor': 0.3,  # More aggressive learning rate reduction
    'lr_patience': 20,  # Reduced patience for faster learning rate adjustment
    'min_lr': 1e-8,  # Lower minimum learning rate for maximum optimization
    'target_accuracy': 0.70,  # Set to 70% as required in problem statement
    'accuracy_patience': 200,  # Increased patience to reach higher target accuracy
    'memory_growth': True,  # Enable memory growth for TensorFlow
    'mixed_precision': False,  # Disable mixed precision for maximum stability
    'gradient_clipping': True,  # Enable gradient clipping
    'max_gradient_norm': 0.5,  # Reduced gradient norm for better stability
    'advanced_optimizers': True,  # Enable advanced optimizer techniques
    'cosine_annealing': True,  # Enable cosine annealing learning rate schedule
    'warm_restarts': True,  # Enable warm restarts for better optimization
    'cyclic_learning_rate': True,  # Enable cyclic learning rate
    'ensemble_training': True,  # Enable ensemble training techniques
    'data_augmentation': True,  # Enable advanced data augmentation
    'regularization_strength': 0.01  # L2 regularization for better generalization
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
