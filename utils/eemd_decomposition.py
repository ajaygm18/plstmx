"""
Ensemble Empirical Mode Decomposition (EEMD) implementation
For noise reduction in stock price time series as per PMC10963254
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging
from config.settings import EEMD_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PyEMD, use fallback if not available
try:
    from PyEMD import EEMD
    PYEMD_AVAILABLE = True
except ImportError:
    logger.warning("PyEMD not available, using fallback implementation")
    PYEMD_AVAILABLE = False
    
    class EEMD:
        def __init__(self, trials=100, noise_width=0.2):
            self.trials = trials
            self.noise_width = noise_width
        
        def eemd(self, signal):
            # Simple fallback: return original signal as the first IMF
            # and noise as the residue
            noise = np.random.normal(0, self.noise_width * np.std(signal), len(signal))
            return [signal, noise]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEMDDecomposer:
    """
    Implements Ensemble Empirical Mode Decomposition for noise reduction
    """
    
    def __init__(self, trials: int = None, noise_width: float = None):
        """
        Initialize EEMD decomposer
        
        Args:
            trials: Number of trials for ensemble
            noise_width: Standard deviation of Gaussian noise
        """
        self.trials = trials or EEMD_CONFIG['trials']
        self.noise_width = noise_width or EEMD_CONFIG['noise_width']
        self.eemd = EEMD(trials=self.trials, noise_width=self.noise_width)
        
    def calculate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """
        Calculate Sample Entropy (SaEn) of a time series
        
        Args:
            data: Input time series
            m: Pattern length
            r: Tolerance for matching (default: 0.2 * std)
            
        Returns:
            Sample entropy value
        """
        if r is None:
            r = 0.2 * np.std(data)
        
        N = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            x = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = x[i]
                for j in range(N - m + 1):
                    if i != j:
                        if _maxdist(template_i, x[j], m) <= r:
                            C[i] += 1.0
            
            phi = np.mean(np.log(C / (N - m)))
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return np.inf  # Return infinity if calculation fails
    
    def decompose_series(self, series: pd.Series) -> Tuple[np.ndarray, List[float], int]:
        """
        Decompose time series using EEMD
        
        Args:
            series: Input time series
            
        Returns:
            Tuple of (IMFs array, sample entropies, highest entropy index)
        """
        try:
            logger.info(f"Starting EEMD decomposition with {self.trials} trials")
            
            # Convert to numpy array
            data = series.values.astype(np.float64)
            
            # Remove any NaN or infinite values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning("Found NaN or infinite values, interpolating...")
                data = pd.Series(data).interpolate().fillna(method='bfill').fillna(method='ffill').values
            
            # Perform EEMD decomposition
            imfs = self.eemd(data)
            
            # Calculate sample entropy for each IMF
            sample_entropies = []
            for i, imf in enumerate(imfs):
                entropy = self.calculate_sample_entropy(imf)
                sample_entropies.append(entropy)
                logger.info(f"IMF {i+1} Sample Entropy: {entropy:.4f}")
            
            # Find IMF with highest sample entropy (most complex/noisy component)
            highest_entropy_idx = np.argmax(sample_entropies)
            
            logger.info(f"Decomposition completed. {len(imfs)} IMFs generated.")
            logger.info(f"Highest entropy IMF: {highest_entropy_idx + 1} (entropy: {sample_entropies[highest_entropy_idx]:.4f})")
            
            return imfs, sample_entropies, highest_entropy_idx
            
        except Exception as e:
            logger.error(f"Error in EEMD decomposition: {str(e)}")
            raise
    
    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, dict]:
        """
        Filter time series by removing the highest entropy IMF component
        
        Args:
            series: Input time series
            
        Returns:
            Tuple of (filtered series, decomposition info)
        """
        try:
            imfs, entropies, highest_entropy_idx = self.decompose_series(series)
            
            # Remove the highest entropy component (most noisy)
            filtered_data = series.values - imfs[highest_entropy_idx]
            filtered_series = pd.Series(filtered_data, index=series.index, name=f"{series.name}_filtered")
            
            # Prepare decomposition info
            decomp_info = {
                'num_imfs': len(imfs),
                'sample_entropies': entropies,
                'removed_imf_index': highest_entropy_idx,
                'removed_imf_entropy': entropies[highest_entropy_idx],
                'noise_reduction_percentage': (np.var(imfs[highest_entropy_idx]) / np.var(series.values)) * 100
            }
            
            logger.info(f"Filtered series created. Noise reduction: {decomp_info['noise_reduction_percentage']:.2f}%")
            
            return filtered_series, decomp_info
            
        except Exception as e:
            logger.error(f"Error in series filtering: {str(e)}")
            raise
    
    def validate_decomposition(self, original: pd.Series, imfs: np.ndarray) -> dict:
        """
        Validate EEMD decomposition by checking reconstruction error
        
        Args:
            original: Original time series
            imfs: Array of IMFs
            
        Returns:
            Validation metrics
        """
        try:
            # Reconstruct signal from IMFs
            reconstructed = np.sum(imfs, axis=0)
            
            # Calculate reconstruction error
            mse = np.mean((original.values - reconstructed) ** 2)
            mae = np.mean(np.abs(original.values - reconstructed))
            correlation = np.corrcoef(original.values, reconstructed)[0, 1]
            
            validation_metrics = {
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'is_valid': mse < 1e-6 and correlation > 0.99  # Tight thresholds for good decomposition
            }
            
            if validation_metrics['is_valid']:
                logger.info("EEMD decomposition validation passed")
            else:
                logger.warning(f"EEMD decomposition validation failed. MSE: {mse}, Correlation: {correlation}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Error in decomposition validation: {str(e)}")
            return {'is_valid': False, 'error': str(e)}

def apply_eemd_filtering(data_dict: dict) -> Tuple[dict, dict]:
    """
    Apply EEMD filtering to all stock indices
    
    Args:
        data_dict: Dictionary mapping index names to DataFrames
        
    Returns:
        Tuple of (filtered_data_dict, decomposition_info_dict)
    """
    decomposer = EEMDDecomposer()
    filtered_data = {}
    decomposition_info = {}
    
    for name, data in data_dict.items():
        try:
            logger.info(f"Applying EEMD filtering to {name}")
            
            # Filter the closing price series
            close_series = data['Close']
            filtered_close, decomp_info = decomposer.filter_series(close_series)
            
            # Create new dataframe with filtered close price
            filtered_df = data.copy()
            filtered_df['Close_Filtered'] = filtered_close
            
            filtered_data[name] = filtered_df
            decomposition_info[name] = decomp_info
            
            logger.info(f"EEMD filtering completed for {name}")
            
        except Exception as e:
            logger.error(f"Failed to apply EEMD filtering for {name}: {str(e)}")
            # If filtering fails, use original data
            filtered_data[name] = data.copy()
            filtered_data[name]['Close_Filtered'] = data['Close']
            decomposition_info[name] = {'error': str(e)}
    
    return filtered_data, decomposition_info

if __name__ == "__main__":
    # Test EEMD decomposition
    from data.data_loader import load_stock_data
    
    data, _ = load_stock_data()
    filtered_data, decomp_info = apply_eemd_filtering(data)
    
    print("EEMD filtering completed!")
    for name, info in decomp_info.items():
        if 'error' not in info:
            print(f"{name}: {info['num_imfs']} IMFs, {info['noise_reduction_percentage']:.2f}% noise reduction")
        else:
            print(f"{name}: Error - {info['error']}")
