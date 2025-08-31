"""
Contractive Autoencoder implementation for feature extraction
As described in PMC10963254 research paper
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict

# Try to import TensorFlow, use fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available, using fallback autoencoder")
    TF_AVAILABLE = False

# Try to import sklearn, use fallback if not available
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn not available, using fallback scaler")
    SKLEARN_AVAILABLE = False
    
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit(self, X):
            # Handle NaN values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.mean_ = np.mean(X_clean, axis=0)
            self.scale_ = np.std(X_clean, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
            return self
        
        def transform(self, X):
            # Handle NaN values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return (X_clean - self.mean_) / self.scale_
        
        def fit_transform(self, X):
            return self.fit(X).transform(X)
        
        def inverse_transform(self, X):
            return X * self.scale_ + self.mean_

from config.settings import CAE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractiveAutoencoder:
    """
    Contractive Autoencoder for dimensionality reduction and feature extraction
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = None, lambda_reg: float = None):
        """
        Initialize Contractive Autoencoder
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Encoded feature dimension
            lambda_reg: Contractive regularization parameter
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or CAE_CONFIG['encoding_dim']
        self.lambda_reg = lambda_reg or CAE_CONFIG['lambda_reg']
        
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self._build_model()
    
    def _build_fallback_model(self):
        """
        Build a fallback model using PCA when TensorFlow is not available
        """
        # Simple PCA-like transformation using SVD
        self.use_fallback = True
        self.encoder = None  # Will be set during training
        logger.info(f"Using PCA fallback: {self.input_dim} -> {self.encoding_dim}")
    
    def _fit_fallback(self, X_scaled: np.ndarray) -> Dict:
        """
        Fallback training using PCA when TensorFlow is not available
        """
        logger.info("Training fallback PCA model...")
        
        # Use SVD for dimensionality reduction
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        
        # Keep top components
        n_components = min(self.encoding_dim, X_scaled.shape[1])
        self.encoder = Vt[:n_components]  # Principal components
        
        self.is_trained = True
        logger.info(f"PCA fallback training completed: {self.input_dim} -> {n_components}")
        
        return {'loss': [0.1], 'val_loss': [0.1]}  # Mock history
    
    def _build_model(self):
        """
        Build the contractive autoencoder architecture
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using PCA fallback for feature extraction")
            self._build_fallback_model()
            return
            
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create models
        self.encoder = keras.Model(input_layer, encoded, name='encoder')
        self.autoencoder = keras.Model(input_layer, decoded, name='contractive_autoencoder')
        
        # Compile with custom loss function
        self.autoencoder.compile(
            optimizer='adam',
            loss=self._contractive_loss,
            metrics=['mse']
        )
        
        logger.info(f"Contractive Autoencoder built: {self.input_dim} -> {self.encoding_dim} -> {self.input_dim}")
    
    def _contractive_loss(self, y_true, y_pred):
        """
        Contractive loss function combining reconstruction loss and contractive penalty
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Combined loss value
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Get the encoded representation
        encoded = self.encoder(y_true)
        
        # Calculate Jacobian penalty (contractive term)
        # This encourages the encoder to be robust to small input variations
        gradients = tf.gradients(encoded, y_true)[0]
        contractive_penalty = tf.reduce_sum(tf.square(gradients), axis=1)
        contractive_penalty = tf.reduce_mean(contractive_penalty)
        
        # Total loss
        total_loss = reconstruction_loss + self.lambda_reg * contractive_penalty
        
        return total_loss
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2, 
            epochs: int = None, batch_size: int = None, verbose: int = 1) -> Dict:
        """
        Train the contractive autoencoder
        
        Args:
            X: Training data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        try:
            epochs = epochs or CAE_CONFIG['epochs']
            batch_size = batch_size or CAE_CONFIG['batch_size']
            
            logger.info(f"Training Contractive Autoencoder for {epochs} epochs")
            
            # Normalize the data and handle NaN values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.fit_transform(X_clean)
            
            # Handle fallback case
            if not TF_AVAILABLE:
                return self._fit_fallback(X_scaled)
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Reduce learning rate on plateau
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
            
            # Train the model
            history = self.autoencoder.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=verbose,
                shuffle=True
            )
            
            self.is_trained = True
            logger.info("Contractive Autoencoder training completed")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training Contractive Autoencoder: {str(e)}")
            raise
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode input data to lower dimensional representation
        
        Args:
            X: Input data
            
        Returns:
            Encoded features
        """
        if not self.is_trained:
            raise ValueError("Autoencoder must be trained before encoding")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Handle fallback case
            if not TF_AVAILABLE:
                # Use PCA transformation
                encoded = X_scaled @ self.encoder.T
                return encoded
            
            encoded = self.encoder.predict(X_scaled, verbose=0)
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding data: {str(e)}")
            raise
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decode encoded data back to original dimension
        
        Args:
            encoded: Encoded data
            
        Returns:
            Reconstructed data
        """
        if not self.is_trained:
            raise ValueError("Autoencoder must be trained before decoding")
        
        try:
            # Create decoder model if not exists
            if not hasattr(self, 'decoder_model'):
                encoded_input = keras.Input(shape=(self.encoding_dim,))
                decoder_layers = self.autoencoder.layers[-4:]  # Last 4 layers are decoder
                x = encoded_input
                for layer in decoder_layers:
                    x = layer(x)
                self.decoder_model = keras.Model(encoded_input, x)
            
            decoded = self.decoder_model.predict(encoded, verbose=0)
            decoded_scaled = self.scaler.inverse_transform(decoded)
            return decoded_scaled
            
        except Exception as e:
            logger.error(f"Error decoding data: {str(e)}")
            raise
    
    def evaluate_reconstruction(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate reconstruction quality
        
        Args:
            X: Test data
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Autoencoder must be trained before evaluation")
        
        try:
            X_scaled = self.scaler.transform(X)
            reconstructed_scaled = self.autoencoder.predict(X_scaled, verbose=0)
            reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
            
            # Calculate metrics
            mse = np.mean((X - reconstructed) ** 2)
            mae = np.mean(np.abs(X - reconstructed))
            
            # Calculate explained variance
            var_original = np.var(X)
            var_error = np.var(X - reconstructed)
            explained_variance = 1 - (var_error / var_original)
            
            # Calculate correlation
            correlation = np.corrcoef(X.flatten(), reconstructed.flatten())[0, 1]
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'explained_variance': explained_variance,
                'correlation': correlation
            }
            
            logger.info(f"Reconstruction evaluation - MSE: {mse:.4f}, Correlation: {correlation:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating reconstruction: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """
        Save the trained autoencoder
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            self.autoencoder.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a trained autoencoder
        
        Args:
            filepath: Path to load the model from
        """
        try:
            self.autoencoder = keras.models.load_model(
                filepath, 
                custom_objects={'_contractive_loss': self._contractive_loss}
            )
            self.encoder = keras.Model(
                self.autoencoder.input, 
                self.autoencoder.get_layer('encoded').output
            )
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def extract_features_with_cae(features_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], Dict[str, ContractiveAutoencoder]]:
    """
    Extract features using Contractive Autoencoder for all indices
    
    Args:
        features_dict: Dictionary mapping index names to feature DataFrames
        
    Returns:
        Tuple of (extracted_features_dict, autoencoder_dict)
    """
    extracted_features = {}
    autoencoders = {}
    
    for name, features_df in features_dict.items():
        try:
            logger.info(f"Training Contractive Autoencoder for {name}")
            
            # Prepare features (remove any remaining NaN values)
            features_clean = features_df.fillna(features_df.mean()).values
            
            # Initialize and train autoencoder
            input_dim = features_clean.shape[1]
            cae = ContractiveAutoencoder(input_dim)
            
            # Train the autoencoder
            history = cae.fit(features_clean)
            
            # Extract features
            encoded_features = cae.encode(features_clean)
            
            # Evaluate reconstruction
            eval_metrics = cae.evaluate_reconstruction(features_clean)
            logger.info(f"CAE for {name} - Explained Variance: {eval_metrics['explained_variance']:.4f}")
            
            extracted_features[name] = encoded_features
            autoencoders[name] = cae
            
        except Exception as e:
            logger.error(f"Failed to extract features for {name}: {str(e)}")
            # If CAE fails, use original features with PCA as fallback
            from sklearn.decomposition import PCA
            pca = PCA(n_components=CAE_CONFIG['encoding_dim'])
            reduced_features = pca.fit_transform(features_df.fillna(features_df.mean()))
            extracted_features[name] = reduced_features
            autoencoders[name] = None
    
    return extracted_features, autoencoders

if __name__ == "__main__":
    # Test Contractive Autoencoder
    np.random.seed(42)
    
    # Generate sample data
    X = np.random.randn(1000, 40)  # 40 features like technical indicators
    
    # Create and train autoencoder
    cae = ContractiveAutoencoder(input_dim=40, encoding_dim=32)
    history = cae.fit(X, epochs=10, verbose=1)
    
    # Test encoding/decoding
    encoded = cae.encode(X)
    metrics = cae.evaluate_reconstruction(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstruction correlation: {metrics['correlation']:.4f}")
