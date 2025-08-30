"""
Data preprocessing module for PLSTM-TAL model
Combines all preprocessing steps including EEMD, technical indicators, and CAE
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from sklearn.model_selection import train_test_split

from data.data_loader import load_stock_data
from data.technical_indicators import calculate_technical_indicators
from utils.eemd_decomposition import apply_eemd_filtering
from utils.contractive_autoencoder import extract_features_with_cae
from config.settings import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, PLSTM_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    """
    Complete preprocessing pipeline for stock market data
    """
    
    def __init__(self):
        self.sequence_length = PLSTM_CONFIG['sequence_length']
        self.data_dict = {}
        self.indicators_dict = {}
        self.filtered_data_dict = {}
        self.features_dict = {}
        self.processed_data = {}
        
    def load_and_prepare_data(self) -> Dict[str, Dict]:
        """
        Load stock data and perform initial preparation
        
        Returns:
            Dictionary with loaded data summary
        """
        try:
            logger.info("Loading stock market data...")
            self.data_dict, summary = load_stock_data()
            
            logger.info("Data loading completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate technical indicators for all indices
        
        Returns:
            Dictionary with technical indicators
        """
        try:
            logger.info("Calculating technical indicators...")
            self.indicators_dict = calculate_technical_indicators(self.data_dict)
            
            logger.info("Technical indicators calculation completed")
            return self.indicators_dict
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
    
    def apply_eemd_filtering(self) -> Dict[str, Dict]:
        """
        Apply EEMD filtering to reduce noise
        
        Returns:
            EEMD decomposition information
        """
        try:
            logger.info("Applying EEMD filtering...")
            self.filtered_data_dict, decomp_info = apply_eemd_filtering(self.data_dict)
            
            logger.info("EEMD filtering completed")
            return decomp_info
            
        except Exception as e:
            logger.error(f"Error applying EEMD filtering: {str(e)}")
            raise
    
    def combine_features(self) -> Dict[str, pd.DataFrame]:
        """
        Combine filtered price data with technical indicators
        
        Returns:
            Combined features for each index
        """
        try:
            logger.info("Combining features...")
            
            for index_name in self.data_dict.keys():
                if index_name in self.filtered_data_dict and index_name in self.indicators_dict:
                    # Get filtered data and indicators
                    filtered_data = self.filtered_data_dict[index_name]
                    indicators = self.indicators_dict[index_name]
                    
                    # Align indices (use intersection)
                    common_index = filtered_data.index.intersection(indicators.index)
                    
                    # Create combined feature set
                    features = pd.DataFrame(index=common_index)
                    
                    # Add original OHLCV data
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in filtered_data.columns:
                            features[col] = filtered_data.loc[common_index, col]
                    
                    # Add filtered close price
                    if 'Close_Filtered' in filtered_data.columns:
                        features['Close_Filtered'] = filtered_data.loc[common_index, 'Close_Filtered']
                    
                    # Add technical indicators
                    for col in indicators.columns:
                        features[f"TI_{col}"] = indicators.loc[common_index, col]
                    
                    # Remove any remaining NaN values
                    features = features.fillna(method='ffill').fillna(method='bfill')
                    
                    self.features_dict[index_name] = features
                    
                    logger.info(f"Combined features for {index_name}: {features.shape}")
                else:
                    logger.warning(f"Missing data for {index_name}, skipping...")
            
            logger.info("Feature combination completed")
            return self.features_dict
            
        except Exception as e:
            logger.error(f"Error combining features: {str(e)}")
            raise
    
    def extract_features_with_autoencoder(self) -> Dict[str, np.ndarray]:
        """
        Extract features using Contractive Autoencoder
        
        Returns:
            Extracted features for each index
        """
        try:
            logger.info("Extracting features with Contractive Autoencoder...")
            
            # Prepare features for CAE (exclude target-related columns)
            cae_features_dict = {}
            
            for index_name, features in self.features_dict.items():
                # Select features for CAE (exclude Close price which is the target)
                feature_cols = [col for col in features.columns if col != 'Close']
                cae_features_dict[index_name] = features[feature_cols]
            
            # Apply CAE
            extracted_features, autoencoders = extract_features_with_cae(cae_features_dict)
            
            logger.info("Feature extraction with CAE completed")
            return extracted_features
            
        except Exception as e:
            logger.error(f"Error extracting features with CAE: {str(e)}")
            raise
    
    def create_targets(self) -> Dict[str, np.ndarray]:
        """
        Create binary classification targets (price movement direction)
        
        Returns:
            Binary targets for each index
        """
        try:
            logger.info("Creating classification targets...")
            targets_dict = {}
            
            for index_name, features in self.features_dict.items():
                close_prices = features['Close'].values
                
                # Calculate price change percentage
                price_changes = np.diff(close_prices) / close_prices[:-1]
                
                # Create binary targets (1 for up, 0 for down)
                targets = (price_changes > 0).astype(int)
                
                # Pad with first value to match length
                targets = np.concatenate([[targets[0]], targets])
                
                targets_dict[index_name] = targets
                
                # Log target distribution
                up_pct = np.mean(targets) * 100
                logger.info(f"{index_name} targets - Up: {up_pct:.1f}%, Down: {100-up_pct:.1f}%")
            
            logger.info("Target creation completed")
            return targets_dict
            
        except Exception as e:
            logger.error(f"Error creating targets: {str(e)}")
            raise
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray, 
                        sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            features: Feature array
            targets: Target array
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (feature_sequences, target_sequences)
        """
        try:
            seq_len = sequence_length or self.sequence_length
            
            X_sequences = []
            y_sequences = []
            
            for i in range(seq_len, len(features)):
                X_sequences.append(features[i-seq_len:i])
                y_sequences.append(targets[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            logger.info(f"Created {len(X_sequences)} sequences of length {seq_len}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  train_split: float = None, val_split: float = None) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature sequences
            y: Target sequences
            train_split: Training split ratio
            val_split: Validation split ratio
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            train_split = train_split or TRAIN_SPLIT
            val_split = val_split or VALIDATION_SPLIT
            
            # Calculate split indices
            n_samples = len(X)
            train_end = int(n_samples * train_split)
            val_end = int(n_samples * (train_split + val_split))
            
            # Split data chronologically (important for time series)
            X_train = X[:train_end]
            X_val = X[train_end:val_end]
            X_test = X[val_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def process_all_indices(self) -> Dict[str, Dict]:
        """
        Process all stock indices through complete pipeline
        
        Returns:
            Processed data for all indices
        """
        try:
            # Step 1: Load data
            summary = self.load_and_prepare_data()
            
            # Step 2: Calculate technical indicators
            self.calculate_indicators()
            
            # Step 3: Apply EEMD filtering
            decomp_info = self.apply_eemd_filtering()
            
            # Step 4: Combine features
            self.combine_features()
            
            # Step 5: Extract features with CAE
            extracted_features = self.extract_features_with_autoencoder()
            
            # Step 6: Create targets
            targets_dict = self.create_targets()
            
            # Step 7: Create sequences and split data for each index
            for index_name in self.features_dict.keys():
                if index_name in extracted_features and index_name in targets_dict:
                    logger.info(f"Processing sequences for {index_name}...")
                    
                    # Get features and targets
                    features = extracted_features[index_name]
                    targets = targets_dict[index_name]
                    
                    # Ensure same length
                    min_len = min(len(features), len(targets))
                    features = features[:min_len]
                    targets = targets[:min_len]
                    
                    # Create sequences
                    X_seq, y_seq = self.create_sequences(features, targets)
                    
                    # Split data
                    X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_seq, y_seq)
                    
                    # Store processed data
                    self.processed_data[index_name] = {
                        'X_train': X_train,
                        'X_val': X_val,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_val': y_val,
                        'y_test': y_test,
                        'feature_dim': features.shape[1],
                        'sequence_length': self.sequence_length,
                        'original_features': self.features_dict[index_name],
                        'decomposition_info': decomp_info.get(index_name, {}),
                        'data_summary': summary.get(index_name, {})
                    }
            
            logger.info("Complete preprocessing pipeline finished")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
    
    def get_preprocessing_summary(self) -> Dict[str, Dict]:
        """
        Get summary of preprocessing results
        
        Returns:
            Summary statistics for each processed index
        """
        summary = {}
        
        for index_name, data in self.processed_data.items():
            summary[index_name] = {
                'train_samples': len(data['X_train']),
                'val_samples': len(data['X_val']),
                'test_samples': len(data['X_test']),
                'feature_dim': data['feature_dim'],
                'sequence_length': data['sequence_length'],
                'train_up_ratio': np.mean(data['y_train']),
                'val_up_ratio': np.mean(data['y_val']),
                'test_up_ratio': np.mean(data['y_test']),
                'noise_reduction': data['decomposition_info'].get('noise_reduction_percentage', 0)
            }
        
        return summary

def preprocess_stock_data() -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Convenience function to run complete preprocessing pipeline
    
    Returns:
        Tuple of (processed_data, preprocessing_summary)
    """
    preprocessor = StockDataPreprocessor()
    processed_data = preprocessor.process_all_indices()
    summary = preprocessor.get_preprocessing_summary()
    
    return processed_data, summary

if __name__ == "__main__":
    # Test preprocessing pipeline
    processed_data, summary = preprocess_stock_data()
    
    print("Preprocessing completed successfully!")
    print("\nSummary:")
    for index_name, stats in summary.items():
        print(f"\n{index_name}:")
        print(f"  Training samples: {stats['train_samples']}")
        print(f"  Validation samples: {stats['val_samples']}")
        print(f"  Test samples: {stats['test_samples']}")
        print(f"  Feature dimension: {stats['feature_dim']}")
        print(f"  Noise reduction: {stats['noise_reduction']:.2f}%")
