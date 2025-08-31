"""
Technical indicators calculation module
Implements all 40 technical indicators as specified in the research paper
Custom implementation without TA-Lib dependency
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from config.settings import TECHNICAL_INDICATORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicatorsCalculator:
    """
    Calculates technical indicators for stock market data
    """
    
    def __init__(self):
        self.indicators = TECHNICAL_INDICATORS
    
    def calculate_sma(self, data: np.ndarray, period: int = 20) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    def calculate_ema(self, data: np.ndarray, period: int = 20) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period).mean().values
    
    def calculate_wma(self, data: np.ndarray, period: int = 20) -> np.ndarray:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return pd.Series(data).rolling(window=period).apply(lambda x: np.average(x, weights=weights), raw=True).values
    
    def calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[np.nan], rsi.values])
    
    def calculate_macd(self, data: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """MACD Line"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        return ema_fast - ema_slow
    
    def calculate_bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: int = 2) -> np.ndarray:
        """Bollinger Bands Middle Line (SMA)"""
        return self.calculate_sma(data, period)
    
    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Stochastic Oscillator %K"""
        lowest_low = pd.Series(low).rolling(window=period).min()
        highest_high = pd.Series(high).rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent.values
    
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = self.calculate_sma(tp, period)
        mean_dev = pd.Series(tp).rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        return cci.values
    
    def calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Williams %R"""
        highest_high = pd.Series(high).rolling(window=period).max()
        lowest_low = pd.Series(low).rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r.values
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for given OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all technical indicators
        """
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            open_price = data['Open'].values
            volume = data['Volume'].values
            
            indicators_data = {}
            
            # Moving Averages
            indicators_data['SMA'] = self.calculate_sma(close, 20)
            indicators_data['EMA'] = self.calculate_ema(close, 20)
            indicators_data['WMA'] = self.calculate_wma(close, 20)
            
            # Double and Triple EMAs
            ema_20 = self.calculate_ema(close, 20)
            indicators_data['DEMA'] = 2 * ema_20 - self.calculate_ema(ema_20, 20)
            indicators_data['TEMA'] = 3 * ema_20 - 3 * self.calculate_ema(ema_20, 20) + self.calculate_ema(self.calculate_ema(ema_20, 20), 20)
            indicators_data['TRIMA'] = self.calculate_sma(self.calculate_sma(close, 10), 10)
            
            # Adaptive Moving Average (simplified KAMA)
            change = np.abs(np.diff(close, 10))
            volatility = pd.Series(np.abs(np.diff(close))).rolling(window=10).sum()
            efficiency_ratio = change / volatility[9:]
            
            # For KAMA, we'll use a simplified approach with fixed alpha
            # In practice, KAMA would use the efficiency ratio to adjust smoothing
            kama_alpha = 0.2  # Simplified approach
            indicators_data['KAMA'] = np.concatenate([np.full(10, np.nan), 
                                                     pd.Series(close[10:]).ewm(alpha=kama_alpha).mean().values])
            
            # MAMA (simplified)
            indicators_data['MAMA'] = self.calculate_ema(close, 5)
            
            # T3 (simplified as triple smoothed EMA)
            t3_ema1 = self.calculate_ema(close, 5)
            t3_ema2 = self.calculate_ema(t3_ema1, 5)
            indicators_data['T3'] = self.calculate_ema(t3_ema2, 5)
            
            # Price Transforms
            indicators_data['MEDPRICE'] = (high + low) / 2
            indicators_data['TYPPRICE'] = (high + low + close) / 3
            indicators_data['WCLPRICE'] = (high + low + 2 * close) / 4
            indicators_data['MIDPRICE'] = (high + low) / 2
            
            # Bollinger Bands
            indicators_data['BBANDS'] = self.calculate_bollinger_bands(close, 20)
            
            # Parabolic SAR (simplified)
            sar = np.zeros_like(close)
            af = 0.02
            for i in range(1, len(close)):
                if i == 1:
                    sar[i] = low[i]
                else:
                    sar[i] = sar[i-1] + af * (high[i-1] - sar[i-1])
            indicators_data['SAR'] = sar
            
            # Volume Indicators
            # Accumulation/Distribution Line
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_multiplier = np.where(high == low, 0, money_flow_multiplier)
            money_flow_volume = money_flow_multiplier * volume
            indicators_data['AD'] = np.cumsum(money_flow_volume)
            
            # A/D Oscillator (simplified)
            ad_sma_fast = self.calculate_sma(indicators_data['AD'], 3)
            ad_sma_slow = self.calculate_sma(indicators_data['AD'], 10)
            indicators_data['ADOSC'] = ad_sma_fast - ad_sma_slow
            
            # On Balance Volume
            obv = np.zeros_like(volume, dtype=float)
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            indicators_data['OBV'] = obv
            
            # Momentum Indicators
            indicators_data['RSI'] = self.calculate_rsi(close, 14)
            indicators_data['MACD'] = self.calculate_macd(close, 12, 26)
            indicators_data['CCI'] = self.calculate_cci(high, low, close, 20)
            indicators_data['WILLR'] = self.calculate_williams_r(high, low, close, 14)
            indicators_data['STOCH'] = self.calculate_stochastic(high, low, close, 14)
            
            # Momentum
            indicators_data['MOM'] = close - np.roll(close, 10)
            
            # Rate of Change
            indicators_data['ROC'] = ((close - np.roll(close, 10)) / np.roll(close, 10)) * 100
            
            # ADX and related indicators (simplified)
            tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            atr = self.calculate_sma(tr, 14)
            
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            indicators_data['PLUS_DI'] = 100 * self.calculate_sma(dm_plus, 14) / atr
            indicators_data['MINUS_DI'] = 100 * self.calculate_sma(dm_minus, 14) / atr
            
            dx = 100 * np.abs(indicators_data['PLUS_DI'] - indicators_data['MINUS_DI']) / \
                 (indicators_data['PLUS_DI'] + indicators_data['MINUS_DI'])
            indicators_data['DX'] = dx
            indicators_data['ADX'] = self.calculate_sma(dx, 14)
            indicators_data['ADXR'] = (indicators_data['ADX'] + np.roll(indicators_data['ADX'], 14)) / 2
            
            # AROON
            aroon_up = np.zeros_like(close)
            aroon_down = np.zeros_like(close)
            period = 14
            for i in range(period, len(close)):
                high_period = high[i-period:i+1]
                low_period = low[i-period:i+1]
                aroon_up[i] = ((period - np.argmax(high_period)) / period) * 100
                aroon_down[i] = ((period - np.argmin(low_period)) / period) * 100
            
            indicators_data['AROON'] = aroon_up
            indicators_data['AROONOSC'] = aroon_up - aroon_down
            
            # Balance of Power
            indicators_data['BOP'] = (close - open_price) / (high - low)
            indicators_data['BOP'] = np.where(high == low, 0, indicators_data['BOP'])
            
            # Money Flow Index
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            # Handle first element (no previous price)
            tp_diff = np.diff(typical_price)
            positive_flow = np.zeros_like(money_flow)
            negative_flow = np.zeros_like(money_flow)
            
            positive_flow[1:] = np.where(tp_diff > 0, money_flow[1:], 0)
            negative_flow[1:] = np.where(tp_diff < 0, money_flow[1:], 0)
            
            positive_flow_sum = pd.Series(positive_flow).rolling(window=14, min_periods=1).sum()
            negative_flow_sum = pd.Series(negative_flow).rolling(window=14, min_periods=1).sum()
            
            # Robust division by zero handling
            mfi_values = np.full_like(close, 50.0)  # Default to 50 when no direction
            valid_mask = (negative_flow_sum > 0)
            
            money_flow_ratio = positive_flow_sum / negative_flow_sum
            mfi_values[valid_mask] = 100 - (100 / (1 + money_flow_ratio[valid_mask]))
            
            # Handle edge cases
            mfi_values = np.clip(mfi_values, 0, 100)
            indicators_data['MFI'] = mfi_values
            
            # Ultimate Oscillator (simplified)
            bp = close - np.minimum(low, np.roll(close, 1))
            tr_uo = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
            
            # Handle first element
            bp[0] = 0
            tr_uo[0] = high[0] - low[0] if high[0] != low[0] else 1.0
            
            # Calculate rolling sums with minimum periods
            tr_sum_7 = pd.Series(tr_uo).rolling(7, min_periods=1).sum()
            tr_sum_14 = pd.Series(tr_uo).rolling(14, min_periods=1).sum()
            tr_sum_28 = pd.Series(tr_uo).rolling(28, min_periods=1).sum()
            
            bp_sum_7 = pd.Series(bp).rolling(7, min_periods=1).sum()
            bp_sum_14 = pd.Series(bp).rolling(14, min_periods=1).sum()
            bp_sum_28 = pd.Series(bp).rolling(28, min_periods=1).sum()
            
            # Robust division with proper handling of zero denominators
            avg7 = np.full_like(close, 0.5)  # Default neutral value
            avg14 = np.full_like(close, 0.5)
            avg28 = np.full_like(close, 0.5)
            
            mask7 = tr_sum_7 > 0
            mask14 = tr_sum_14 > 0
            mask28 = tr_sum_28 > 0
            
            avg7[mask7] = bp_sum_7[mask7] / tr_sum_7[mask7]
            avg14[mask14] = bp_sum_14[mask14] / tr_sum_14[mask14]
            avg28[mask28] = bp_sum_28[mask28] / tr_sum_28[mask28]
            
            ultosc_values = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            indicators_data['ULTOSC'] = np.clip(ultosc_values, 0, 100)
            
            # Absolute Price Oscillator
            ema_fast = self.calculate_ema(close, 12)
            ema_slow = self.calculate_ema(close, 26)
            indicators_data['APO'] = ema_fast - ema_slow
            
            # Percentage Price Oscillator
            indicators_data['PPO'] = ((ema_fast - ema_slow) / ema_slow) * 100
            
            # Chande Momentum Oscillator
            price_diff = np.diff(close)
            gains = np.where(price_diff > 0, price_diff, 0)
            losses = np.where(price_diff < 0, -price_diff, 0)
            
            sum_gains = pd.Series(gains).rolling(window=14).sum()
            sum_losses = pd.Series(losses).rolling(window=14).sum()
            
            cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
            indicators_data['CMO'] = np.concatenate([[np.nan], cmo.values])
            
            # Stochastic RSI
            rsi_values = indicators_data['RSI'].copy()
            
            # Handle NaN values in RSI first
            rsi_values = pd.Series(rsi_values).fillna(50).values  # Fill NaN with neutral RSI
            
            rsi_low = pd.Series(rsi_values).rolling(window=14, min_periods=1).min()
            rsi_high = pd.Series(rsi_values).rolling(window=14, min_periods=1).max()
            rsi_range = rsi_high - rsi_low
            
            # Initialize with neutral values
            stoch_rsi = np.full_like(rsi_values, 50.0)
            
            # Only calculate when we have a valid range
            valid_mask = (rsi_range > 0)
            stoch_rsi[valid_mask] = ((rsi_values[valid_mask] - rsi_low[valid_mask]) / 
                                   rsi_range[valid_mask]) * 100
            
            # Ensure values are within bounds
            stoch_rsi = np.clip(stoch_rsi, 0, 100)
            indicators_data['STOCHRSI'] = stoch_rsi
            
            # Log Return
            indicators_data['LOG_RETURN'] = np.log(close / np.roll(close, 1))
            indicators_data['LOG_RETURN'][0] = 0
            
            # Create DataFrame
            indicators_df = pd.DataFrame(indicators_data, index=data.index)
            
            # Handle NaN values by forward filling and then backward filling
            indicators_df = indicators_df.ffill().bfill()
            
            logger.info(f"Calculated {len(indicators_df.columns)} technical indicators")
            return indicators_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def get_indicator_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all technical indicators
        
        Returns:
            Dictionary mapping indicator names to descriptions
        """
        descriptions = {
            'BBANDS': 'Bollinger Bands - volatility indicator',
            'WMA': 'Weighted Moving Average',
            'EMA': 'Exponential Moving Average',
            'DEMA': 'Double Exponential Moving Average',
            'KAMA': 'Kaufman Adaptive Moving Average',
            'MAMA': 'MESA Adaptive Moving Average',
            'MIDPRICE': 'Midpoint Price over period',
            'SAR': 'Parabolic Stop and Reverse',
            'SMA': 'Simple Moving Average',
            'T3': 'Triple Exponential Moving Average',
            'TEMA': 'Triple Exponential Moving Average',
            'TRIMA': 'Triangular Moving Average',
            'AD': 'Accumulation/Distribution Line',
            'ADOSC': 'Accumulation/Distribution Oscillator',
            'OBV': 'On Balance Volume',
            'MEDPRICE': 'Median Price',
            'TYPPRICE': 'Typical Price',
            'WCLPRICE': 'Weighted Close Price',
            'ADX': 'Average Directional Movement Index',
            'ADXR': 'Average Directional Movement Index Rating',
            'APO': 'Absolute Price Oscillator',
            'AROON': 'Aroon',
            'AROONOSC': 'Aroon Oscillator',
            'BOP': 'Balance Of Power',
            'CCI': 'Commodity Channel Index',
            'CMO': 'Chande Momentum Oscillator',
            'DX': 'Directional Movement Index',
            'MACD': 'Moving Average Convergence/Divergence',
            'MFI': 'Money Flow Index',
            'MINUS_DI': 'Minus Directional Indicator',
            'MOM': 'Momentum',
            'PLUS_DI': 'Plus Directional Indicator',
            'LOG_RETURN': 'Logarithmic Return',
            'PPO': 'Percentage Price Oscillator',
            'ROC': 'Rate of change',
            'RSI': 'Relative Strength Index',
            'STOCH': 'Stochastic',
            'STOCHRSI': 'Stochastic Relative Strength Index',
            'ULTOSC': 'Ultimate Oscillator',
            'WILLR': 'Williams %R'
        }
        return descriptions
    
    def validate_indicators(self, indicators_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate calculated indicators
        
        Args:
            indicators_df: DataFrame with calculated indicators
            
        Returns:
            Dictionary mapping indicator names to validation status
        """
        validation_results = {}
        
        for indicator in indicators_df.columns:
            is_valid = True
            issues = []
            
            # Check for all NaN values
            if indicators_df[indicator].isna().all():
                issues.append("All values are NaN")
                is_valid = False
            
            # Check for infinite values
            if np.isinf(indicators_df[indicator]).any():
                issues.append("Contains infinite values")
                is_valid = False
            
            # Check for excessive NaN values (more than 20% for better threshold)
            nan_percentage = indicators_df[indicator].isna().mean()
            if nan_percentage > 0.2:
                issues.append(f"High NaN percentage: {nan_percentage:.2%}")
                is_valid = False
            
            # Check for constant values (might indicate calculation issues)
            if is_valid:
                non_nan_values = indicators_df[indicator].dropna()
                if len(non_nan_values) > 1 and non_nan_values.nunique() == 1:
                    issues.append("All non-NaN values are identical")
                    # Don't mark as invalid, but warn
            
            # Check for reasonable value ranges for specific indicators
            if is_valid and indicator in ['RSI', 'STOCHRSI', 'MFI', 'ULTOSC']:
                values = indicators_df[indicator].dropna()
                if len(values) > 0:
                    if values.min() < -1 or values.max() > 101:
                        issues.append(f"Values outside expected range [0,100]: [{values.min():.2f}, {values.max():.2f}]")
            
            validation_results[indicator] = {
                'is_valid': is_valid,
                'issues': issues,
                'nan_percentage': nan_percentage,
                'min_value': indicators_df[indicator].min() if not indicators_df[indicator].isna().all() else np.nan,
                'max_value': indicators_df[indicator].max() if not indicators_df[indicator].isna().all() else np.nan
            }
            
            if issues:
                logger.warning(f"Validation issues for {indicator}: {', '.join(issues)}")
        
        return validation_results

def calculate_technical_indicators(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate technical indicators for all stock indices
    
    Args:
        data_dict: Dictionary mapping index names to OHLCV DataFrames
        
    Returns:
        Dictionary mapping index names to technical indicators DataFrames
    """
    calculator = TechnicalIndicatorsCalculator()
    indicators_dict = {}
    
    for name, data in data_dict.items():
        try:
            logger.info(f"Calculating technical indicators for {name}")
            indicators = calculator.calculate_indicators(data)
            validation = calculator.validate_indicators(indicators)
            
            # Log validation summary
            valid_count = sum(1 for v in validation.values() if v['is_valid'])
            total_count = len(validation)
            logger.info(f"Technical indicators for {name}: {valid_count}/{total_count} passed validation")
            
            indicators_dict[name] = indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators for {name}: {str(e)}")
            continue
    
    return indicators_dict

if __name__ == "__main__":
    # Test technical indicators calculation
    from data.data_loader import load_stock_data
    
    data, _ = load_stock_data()
    indicators = calculate_technical_indicators(data)
    
    print("Technical indicators calculation completed!")
    for name, indicators_df in indicators.items():
        print(f"{name}: {len(indicators_df.columns)} indicators calculated")
