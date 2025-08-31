"""
PLSTM-TAL (Peephole LSTM with Temporal Attention Layer) model implementation
Based on the research paper PMC10963254
"""

import numpy as np
import pandas as pd
import logging
import os
import time
from typing import Tuple, Dict, Optional

# Configure GPU settings before TensorFlow import
try:
    from utils.gpu_config import ensure_gpu_configured
    ensure_gpu_configured()
except ImportError:
    # gpu_config module not available, skip GPU configuration
    pass

# Try to import TensorFlow, use fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available, using fallback model")
    TF_AVAILABLE = False

# Try to import sklearn, use fallback if not available
try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn not available, using fallback scaler")
    SKLEARN_AVAILABLE = False
    
    class MinMaxScaler:
        def __init__(self):
            self.min_ = None
            self.scale_ = None
        
        def fit(self, X):
            self.min_ = np.min(X, axis=0)
            self.scale_ = np.max(X, axis=0) - self.min_
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
            return self
        
        def transform(self, X):
            return (X - self.min_) / self.scale_
        
        def fit_transform(self, X):
            return self.fit(X).transform(X)
        
        def inverse_transform(self, X):
            return X * self.scale_ + self.min_

from config.settings import PLSTM_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback classes when TensorFlow is not available
if not TF_AVAILABLE:
    class Layer:
        def __init__(self, **kwargs):
            pass
        
        def build(self, input_shape):
            pass
        
        def call(self, inputs):
            return inputs
    
    class Model:
        def __init__(self, *args, **kwargs):
            self.weights = []
        
        def compile(self, *args, **kwargs):
            pass
        
        def fit(self, *args, **kwargs):
            return {'loss': [0.1], 'accuracy': [0.5]}
        
        def predict(self, X):
            # Return random predictions with correct shape
            if len(X.shape) == 3:  # sequence data
                return np.random.random((X.shape[0], 1))
            return np.random.random((X.shape[0], 1))
        
        def evaluate(self, X, y):
            return {'loss': 0.1, 'accuracy': 0.5}
    
    layers = type('layers', (), {
        'Layer': Layer,
        'Dense': Layer,
        'Dropout': Layer,
        'RNN': Layer,
        'Input': lambda shape: None
    })()

class TemporalAttentionLayer(layers.Layer if TF_AVAILABLE else Layer):
    """
    Temporal Attention Layer for enhanced temporal pattern recognition
    """
    
    def __init__(self, attention_units: int, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units
        
    def build(self, input_shape):
        if not TF_AVAILABLE:
            return
            
        # Attention weights
        self.W_a = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1], self.attention_units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.U_a = self.add_weight(
            name='attention_context',
            shape=(self.attention_units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_a = self.add_weight(
            name='attention_bias',
            shape=(self.attention_units,),
            initializer='zeros',
            trainable=True
        )
        super(TemporalAttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        if not TF_AVAILABLE:
            # Fallback: simple mean pooling over time dimension
            if len(inputs.shape) == 3:
                context_vector = np.mean(inputs, axis=1)
                attention_weights = np.ones((inputs.shape[0], inputs.shape[1], 1)) / inputs.shape[1]
            else:
                context_vector = inputs
                attention_weights = np.ones((inputs.shape[0], 1, 1))
            return context_vector, attention_weights
            
        # inputs shape: (batch_size, time_steps, features)
        # Compute attention scores
        attention_scores = tf.nn.tanh(tf.tensordot(inputs, self.W_a, axes=1) + self.b_a)
        attention_scores = tf.tensordot(attention_scores, self.U_a, axes=1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention weights to input
        attended_output = inputs * attention_weights
        
        # Sum over time dimension to get final output
        context_vector = tf.reduce_sum(attended_output, axis=1)
        
        return context_vector, attention_weights
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1], 1)]
    
    def get_config(self):
        config = super(TemporalAttentionLayer, self).get_config()
        config.update({'attention_units': self.attention_units})
        return config

class PeepholeLSTMCell(layers.Layer if TF_AVAILABLE else Layer):
    """
    Custom Peephole LSTM Cell with peephole connections
    """
    
    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid',
                 dropout=0.0, recurrent_dropout=0.0, **kwargs):
        super(PeepholeLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        
        # Required for RNN compatibility
        self.state_size = [self.units, self.units]  # [h_state, c_state]
        self.output_size = int(self.units)  # Ensure it's a Python int
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Input to hidden weights
        self.W_i = self.add_weight(shape=(input_dim, self.units), name='W_i', initializer='glorot_uniform')
        self.W_f = self.add_weight(shape=(input_dim, self.units), name='W_f', initializer='glorot_uniform')
        self.W_c = self.add_weight(shape=(input_dim, self.units), name='W_c', initializer='glorot_uniform')
        self.W_o = self.add_weight(shape=(input_dim, self.units), name='W_o', initializer='glorot_uniform')
        
        # Hidden to hidden weights
        self.U_i = self.add_weight(shape=(self.units, self.units), name='U_i', initializer='orthogonal')
        self.U_f = self.add_weight(shape=(self.units, self.units), name='U_f', initializer='orthogonal')
        self.U_c = self.add_weight(shape=(self.units, self.units), name='U_c', initializer='orthogonal')
        self.U_o = self.add_weight(shape=(self.units, self.units), name='U_o', initializer='orthogonal')
        
        # Peephole connections (cell state to gates)
        self.V_i = self.add_weight(shape=(self.units,), name='V_i', initializer='ones')
        self.V_f = self.add_weight(shape=(self.units,), name='V_f', initializer='ones')
        self.V_o = self.add_weight(shape=(self.units,), name='V_o', initializer='ones')
        
        # Biases
        self.b_i = self.add_weight(shape=(self.units,), name='b_i', initializer='zeros')
        self.b_f = self.add_weight(shape=(self.units,), name='b_f', initializer='ones')
        self.b_c = self.add_weight(shape=(self.units,), name='b_c', initializer='zeros')
        self.b_o = self.add_weight(shape=(self.units,), name='b_o', initializer='zeros')
        
        super(PeepholeLSTMCell, self).build(input_shape)
    
    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        
        # Input gate with peephole connection
        i_t = self.recurrent_activation(
            tf.matmul(inputs, self.W_i) + 
            tf.matmul(h_tm1, self.U_i) + 
            self.V_i * c_tm1 + self.b_i
        )
        
        # Forget gate with peephole connection
        f_t = self.recurrent_activation(
            tf.matmul(inputs, self.W_f) + 
            tf.matmul(h_tm1, self.U_f) + 
            self.V_f * c_tm1 + self.b_f
        )
        
        # Candidate values
        c_tilde = self.activation(
            tf.matmul(inputs, self.W_c) + 
            tf.matmul(h_tm1, self.U_c) + self.b_c
        )
        
        # Cell state
        c_t = f_t * c_tm1 + i_t * c_tilde
        
        # Output gate with peephole connection
        o_t = self.recurrent_activation(
            tf.matmul(inputs, self.W_o) + 
            tf.matmul(h_tm1, self.U_o) + 
            self.V_o * c_t + self.b_o
        )
        
        # Hidden state
        h_t = o_t * self.activation(c_t)
        
        return h_t, [h_t, c_t]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.units), dtype=dtype),
                tf.zeros((batch_size, self.units), dtype=dtype)]

class PLSTMTAL:
    """
    Peephole LSTM with Temporal Attention Layer (PLSTM-TAL) model
    """
    
    def __init__(self, sequence_length: int = None, n_features: int = None, 
                 lstm_units: int = None, attention_units: int = None,
                 dropout_rate: float = None):
        """
        Initialize PLSTM-TAL model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            lstm_units: Number of LSTM units
            attention_units: Number of attention units
            dropout_rate: Dropout rate
        """
        self.sequence_length = sequence_length or PLSTM_CONFIG['sequence_length']
        self.n_features = n_features
        self.lstm_units = lstm_units or PLSTM_CONFIG['lstm_units']
        self.attention_units = attention_units or PLSTM_CONFIG['attention_units']
        self.dropout_rate = dropout_rate or PLSTM_CONFIG['dropout_rate']
        
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self) -> Model:
        """
        Build the PLSTM-TAL model architecture
        
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using fallback model")
            self.model = Model()  # Use fallback Model class
            return self.model
            
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Peephole LSTM layers
        lstm_cell = PeepholeLSTMCell(self.lstm_units, dropout=self.dropout_rate)
        lstm_layer = layers.RNN(lstm_cell, return_sequences=True, return_state=False)
        lstm_output = lstm_layer(inputs)
        
        # Add another Peephole LSTM layer for deeper representation
        lstm_cell2 = PeepholeLSTMCell(self.lstm_units // 2, dropout=self.dropout_rate)
        lstm_layer2 = layers.RNN(lstm_cell2, return_sequences=True, return_state=False)
        lstm_output = lstm_layer2(lstm_output)
        
        # Temporal Attention Layer
        attention_layer = TemporalAttentionLayer(self.attention_units)
        attended_output, attention_weights = attention_layer(lstm_output)
        
        # Dense layers for classification
        dense1 = layers.Dense(64, activation='relu')(attended_output)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense2 = layers.Dropout(self.dropout_rate)(dense2)
        
        # Output layer (binary classification - up/down movement)
        output = layers.Dense(1, activation='sigmoid', name='prediction')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=output, name='PLSTM_TAL')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=PLSTM_CONFIG['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info("PLSTM-TAL model built successfully")
        logger.info(f"Model summary: {model.count_params()} parameters")
        
        return model
    
    def prepare_data(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper scaling and sequencing
        
        Args:
            features: Feature array
            targets: Target array (binary classification)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        try:
            # Scale features
            features_scaled = self.scaler_X.fit_transform(features)
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(self.sequence_length, len(features_scaled)):
                X_sequences.append(features_scaled[i-self.sequence_length:i])
                y_sequences.append(targets[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            logger.info(f"Prepared {len(X_sequences)} sequences")
            logger.info(f"X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_data: tuple = None, 
              validation_split: float = 0.2, epochs: int = None, batch_size: int = None, 
              verbose: int = 1, patience: int = None, min_delta: float = None,
              restore_best_weights: bool = None, monitor: str = None, mode: str = None,
              save_best_only: bool = None, **kwargs) -> Dict:
        """
        Enhanced train method for PLSTM-TAL model with extended training support
        
        Args:
            X: Feature sequences
            y: Target sequences
            validation_data: Tuple of (X_val, y_val) for validation
            validation_split: Fraction for validation (if validation_data not provided)
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            restore_best_weights: Whether to restore best weights
            monitor: Metric to monitor for callbacks
            mode: Mode for monitoring (min/max)
            save_best_only: Whether to save only best model
            **kwargs: Additional training parameters
            
        Returns:
            Enhanced training history
        """
        try:
            # Import extended configuration
            from config.settings import EXTENDED_TRAINING_CONFIG
            
            # Use enhanced defaults
            epochs = epochs or EXTENDED_TRAINING_CONFIG.get('epochs', PLSTM_CONFIG['epochs'])
            batch_size = batch_size or PLSTM_CONFIG['batch_size']
            patience = patience or EXTENDED_TRAINING_CONFIG.get('patience', 50)
            min_delta = min_delta or EXTENDED_TRAINING_CONFIG.get('min_delta', 0.0001)
            restore_best_weights = restore_best_weights if restore_best_weights is not None else EXTENDED_TRAINING_CONFIG.get('restore_best_weights', True)
            monitor = monitor or EXTENDED_TRAINING_CONFIG.get('monitor', 'val_accuracy')
            mode = mode or EXTENDED_TRAINING_CONFIG.get('mode', 'max')
            save_best_only = save_best_only if save_best_only is not None else EXTENDED_TRAINING_CONFIG.get('save_best_only', True)
            
            if self.model is None:
                self.n_features = X.shape[2]
                self.build_model()
            
            logger.info(f"Starting enhanced PLSTM-TAL training for {epochs} epochs")
            logger.info(f"Target accuracy: {EXTENDED_TRAINING_CONFIG.get('target_accuracy', 0.7):.1%}")
            logger.info(f"Monitor: {monitor}, Mode: {mode}, Patience: {patience}")
            
            # Enhanced callbacks for extended training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=restore_best_weights,
                    verbose=verbose
                )
            ]
            
            # Add learning rate reduction if enabled
            if EXTENDED_TRAINING_CONFIG.get('reduce_lr_on_plateau', True):
                callbacks.append(
                    keras.callbacks.ReduceLROnPlateau(
                        monitor=monitor,
                        factor=EXTENDED_TRAINING_CONFIG.get('lr_reduction_factor', 0.5),
                        patience=EXTENDED_TRAINING_CONFIG.get('lr_patience', 30),
                        min_lr=EXTENDED_TRAINING_CONFIG.get('min_lr', 1e-7),
                        verbose=verbose
                    )
                )
            
            # Add model checkpointing
            checkpoint_path = os.path.join(MODELS_DIR, f'plstm_tal_best_{int(time.time())}.h5')
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor=monitor,
                    save_best_only=save_best_only,
                    mode=mode,
                    save_weights_only=False,
                    verbose=verbose
                )
            )
            
            # Add TensorBoard logging if enabled
            if EXTENDED_TRAINING_CONFIG.get('enable_tensorboard', True):
                log_dir = os.path.join(EXTENDED_TRAINING_CONFIG.get('log_dir', 'logs'), 
                                     f'plstm_tal_{int(time.time())}')
                callbacks.append(
                    keras.callbacks.TensorBoard(
                        log_dir=log_dir,
                        histogram_freq=1,
                        write_graph=True,
                        write_images=True,
                        update_freq='epoch'
                    )
                )
            
            # Custom callback to check target accuracy
            class TargetAccuracyCallback(keras.callbacks.Callback):
                def __init__(self, target_accuracy):
                    super().__init__()
                    self.target_accuracy = target_accuracy
                    
                def on_epoch_end(self, epoch, logs=None):
                    val_acc = logs.get('val_accuracy', 0)
                    if val_acc >= self.target_accuracy:
                        logger.info(f"🎯 Target accuracy {self.target_accuracy:.1%} achieved at epoch {epoch + 1}!")
                        logger.info(f"Current validation accuracy: {val_acc:.4f}")
            
            callbacks.append(TargetAccuracyCallback(EXTENDED_TRAINING_CONFIG.get('target_accuracy', 0.7)))
            
            # Prepare training arguments
            train_args = {
                'x': X,
                'y': y,
                'epochs': epochs,
                'batch_size': batch_size,
                'callbacks': callbacks,
                'verbose': verbose,
                'shuffle': True
            }
            
            # Use validation_data if provided, otherwise use validation_split
            if validation_data is not None:
                train_args['validation_data'] = validation_data
            else:
                train_args['validation_split'] = validation_split
            
            logger.info("Starting training with enhanced configuration...")
            start_time = time.time()
            
            # Train the model with no timeout
            history = self.model.fit(**train_args)
            
            training_time = time.time() - start_time
            
            self.is_trained = True
            
            # Enhanced training completion logging
            best_val_acc = max(history.history.get('val_accuracy', [0]))
            target_acc = EXTENDED_TRAINING_CONFIG.get('target_accuracy', 0.7)
            
            logger.info(f"Enhanced PLSTM-TAL training completed in {training_time:.2f}s")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"Target accuracy ({target_acc:.1%}) achieved: {best_val_acc >= target_acc}")
            
            # Add training metadata to history
            if hasattr(history, 'history'):
                history.history['training_time'] = training_time
                history.history['target_accuracy'] = target_acc
                history.history['target_achieved'] = best_val_acc >= target_acc
                history.history['best_val_accuracy'] = best_val_acc
            
            return history
            
        except Exception as e:
            logger.error(f"Error in enhanced PLSTM-TAL training: {str(e)}")
            raise
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training PLSTM-TAL: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            predictions = self.model.predict(X, verbose=0)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            # Get predictions
            y_pred_prob = self.predict(X)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate metrics
            loss, accuracy, precision, recall = self.model.evaluate(X, y, verbose=0)
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        try:
            custom_objects = {
                'TemporalAttentionLayer': TemporalAttentionLayer,
                'PeepholeLSTMCell': PeepholeLSTMCell
            }
            self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    # Test PLSTM-TAL model
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    n_features = 32
    sequence_length = 60
    
    # Create dummy features and targets
    features = np.random.randn(n_samples, n_features)
    targets = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Initialize model
    model = PLSTMTAL(sequence_length=sequence_length, n_features=n_features)
    
    # Prepare data
    X_seq, y_seq = model.prepare_data(features, targets)
    
    # Build and train model
    model.build_model()
    history = model.train(X_seq, y_seq, epochs=5, verbose=1)
    
    # Evaluate model
    metrics = model.evaluate(X_seq, y_seq)
    
    print(f"Test completed - Accuracy: {metrics['accuracy']:.4f}")
