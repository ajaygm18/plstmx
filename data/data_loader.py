"""
Data loader module for stock market data collection
Implements data collection for 4 stock indices as per PMC10963254
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from config.settings import STOCK_INDICES, START_DATE, END_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    Loads stock market data for multiple indices
    """
    
    def __init__(self):
        self.indices = STOCK_INDICES
        self.start_date = START_DATE
        self.end_date = END_DATE
        self.data = {}
    
    def load_single_index(self, symbol: str, name: str) -> pd.DataFrame:
        """
        Load data for a single stock index
        
        Args:
            symbol: Stock symbol (e.g., '^GSPC')
            name: Index name (e.g., 'US')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading data for {name} ({symbol})")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns for {symbol}")
            
            # Clean data
            data = data.dropna()
            data = data[required_columns]
            
            logger.info(f"Loaded {len(data)} records for {name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {name} ({symbol}): {str(e)}")
            raise
    
    def load_all_indices(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for all stock indices
        
        Returns:
            Dictionary mapping index names to DataFrames
        """
        all_data = {}
        
        for name, symbol in self.indices.items():
            try:
                data = self.load_single_index(symbol, name)
                all_data[name] = data
            except Exception as e:
                logger.error(f"Failed to load {name}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("Failed to load any stock data")
        
        self.data = all_data
        return all_data
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for loaded data
        
        Returns:
            Dictionary with summary stats for each index
        """
        summary = {}
        
        for name, data in self.data.items():
            summary[name] = {
                'records': len(data),
                'start_date': data.index.min().strftime('%Y-%m-%d'),
                'end_date': data.index.max().strftime('%Y-%m-%d'),
                'mean_close': data['Close'].mean(),
                'std_close': data['Close'].std(),
                'min_close': data['Close'].min(),
                'max_close': data['Close'].max()
            }
        
        return summary
    
    def validate_data_quality(self) -> Dict[str, bool]:
        """
        Validate data quality for each index
        
        Returns:
            Dictionary mapping index names to validation status
        """
        validation_results = {}
        
        for name, data in self.data.items():
            is_valid = True
            issues = []
            
            # Check for sufficient data points
            if len(data) < 252:  # Less than 1 year of trading days
                issues.append("Insufficient data points")
                is_valid = False
            
            # Check for missing values
            if data.isnull().any().any():
                issues.append("Contains missing values")
                is_valid = False
            
            # Check for negative values in price/volume
            if (data[['Open', 'High', 'Low', 'Close', 'Volume']] < 0).any().any():
                issues.append("Contains negative values")
                is_valid = False
            
            # Check for zero volume days (too many might indicate data issues)
            zero_volume_pct = (data['Volume'] == 0).mean()
            if zero_volume_pct > 0.05:  # More than 5% zero volume days
                issues.append(f"High percentage of zero volume days: {zero_volume_pct:.2%}")
                is_valid = False
            
            validation_results[name] = {
                'is_valid': is_valid,
                'issues': issues
            }
            
            if issues:
                logger.warning(f"Data quality issues for {name}: {', '.join(issues)}")
        
        return validation_results

def load_stock_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Convenience function to load and validate stock data
    
    Returns:
        Tuple of (data_dict, summary_dict)
    """
    loader = StockDataLoader()
    data = loader.load_all_indices()
    summary = loader.get_data_summary()
    validation = loader.validate_data_quality()
    
    # Log validation results
    for name, result in validation.items():
        if result['is_valid']:
            logger.info(f"Data validation passed for {name}")
        else:
            logger.warning(f"Data validation failed for {name}: {result['issues']}")
    
    return data, summary

if __name__ == "__main__":
    # Test data loading
    data, summary = load_stock_data()
    print("Data loading completed successfully!")
    for name, stats in summary.items():
        print(f"{name}: {stats['records']} records from {stats['start_date']} to {stats['end_date']}")
