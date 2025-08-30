"""
PLSTM-TAL (Peephole LSTM with Temporal Attention Layer) model implementation
Based on the research paper PMC10963254
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional
import logging
from config.settings import PLSTM_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalAttentionLayer(layers.Layer):
    """
    Temporal Attention Layer for enhanced temporal pattern recognition
    """
    
    def __init__(self, attention_units: int, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units
        
    def build(self, input_shape):
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

class PeepholeLSTMCell(layers.Layer):
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
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = None, batch_size: int = None, verbose: int = 1) -> Dict:
        """
        Train the PLSTM-TAL model
        
        Args:
            X: Feature sequences
            y: Target sequences
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        try:
            epochs = epochs or PLSTM_CONFIG['epochs']
            batch_size = batch_size or PLSTM_CONFIG['batch_size']
            
            if self.model is None:
                self.n_features = X.shape[2]
                self.build_model()
            
            logger.info(f"Training PLSTM-TAL for {epochs} epochs")
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_plstm_tal.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
            
            # Train the model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )
            
            self.is_trained = True
            logger.info("PLSTM-TAL training completed")
            
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
