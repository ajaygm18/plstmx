"""
Mock implementations for missing dependencies
This allows the application to run without installing packages
"""

import sys
import os
from types import ModuleType

# Check if real dependencies are available
def check_real_dependencies():
    """Check if real dependencies are available and skip mocks if they are"""
    real_deps_available = True
    
    try:
        import tensorflow
        import numpy  
        import pandas
        import sklearn
        import yfinance
        from PyEMD import EEMD
        print("✅ Real dependencies detected - skipping mock initialization")
        return True
    except ImportError as e:
        print(f"⚠️  Missing dependencies, loading mocks: {e}")
        return False

# Only install mocks if real dependencies are not available
if check_real_dependencies():
    # Real dependencies available, exit early
    sys.exit(0)

class MockArray:
    def __init__(self, data=None, shape=None):
        if data is not None:
            if hasattr(data, '__len__'):
                self.data = list(data)
                self.shape = (len(data),)
            else:
                self.data = [data]
                self.shape = (1,)
        else:
            self.data = []
            self.shape = shape or (0,)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockArray(self.data[key])
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def flatten(self):
        return MockArray(self.data)
    
    def astype(self, dtype):
        return MockArray(self.data)
    
    def mean(self, axis=None):
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)
    
    def std(self, axis=None):
        if not self.data:
            return 0
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5
    
    def min(self):
        return min(self.data) if self.data else 0
    
    def max(self):
        return max(self.data) if self.data else 0
    
    def sum(self):
        return sum(self.data) if self.data else 0
    
    # Make it behave more like numpy array for calculations
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x + other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x + y for x, y in zip(self.data, other.data)])
        return MockArray([x + other for x in self.data])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x - other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x - y for x, y in zip(self.data, other.data)])
        return MockArray([x - other for x in self.data])
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([other - x for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([y - x for x, y in zip(self.data, other.data)])
        return MockArray([other - x for x in self.data])
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x * other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x * y for x, y in zip(self.data, other.data)])
        return MockArray([x * other for x in self.data])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x / other if other != 0 else 0 for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x / y if y != 0 else 0 for x, y in zip(self.data, other.data)])
        return MockArray([x / other if other != 0 else 0 for x in self.data])
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([other / x if x != 0 else 0 for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([y / x if x != 0 else 0 for x, y in zip(self.data, other.data)])
        return MockArray([other / x if x != 0 else 0 for x in self.data])
    
    # Comparison operations
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x == other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x == y for x, y in zip(self.data, other.data)])
        return MockArray([x == other for x in self.data])
    
    def __ne__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x != other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x != y for x, y in zip(self.data, other.data)])
        return MockArray([x != other for x in self.data])
    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x < other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x < y for x, y in zip(self.data, other.data)])
        return MockArray([x < other for x in self.data])
    
    def __le__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x <= other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x <= y for x, y in zip(self.data, other.data)])
        return MockArray([x <= other for x in self.data])
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x > other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x > y for x, y in zip(self.data, other.data)])
        return MockArray([x > other for x in self.data])
    
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return MockArray([x >= other for x in self.data])
        elif hasattr(other, 'data'):
            return MockArray([x >= y for x, y in zip(self.data, other.data)])
        return MockArray([x >= other for x in self.data])
    
    # Unary operations
    def __neg__(self):
        """Unary negation"""
        return MockArray([-x for x in self.data])
    
    def __pos__(self):
        """Unary plus"""
        return MockArray([+x for x in self.data])
    
    def __abs__(self):
        """Absolute value"""
        return MockArray([abs(x) for x in self.data])
    
    # Make it iterable
    def __iter__(self):
        return iter(self.data)
    
    # Support numpy-style operations
    @property
    def values(self):
        return self.data

class MockNumpy:
    array = MockArray
    ndarray = MockArray
    float64 = float  # Add float64 type
    
    @staticmethod
    def random(*args, **kwargs):
        return MockArray([0.5] * 10)
    
    @staticmethod
    def zeros(shape):
        if isinstance(shape, (list, tuple)):
            size = 1
            for dim in shape:
                size *= dim
        else:
            size = shape
        return MockArray([0] * size, shape)
    
    @staticmethod
    def zeros_like(arr, dtype=None):
        """Create array of zeros with same shape as input"""
        if hasattr(arr, 'data'):
            size = len(arr.data)
            shape = arr.shape
        else:
            size = len(arr)
            shape = (size,)
        return MockArray([0] * size, shape)
    
    @staticmethod
    def ones(shape):
        if isinstance(shape, (list, tuple)):
            size = 1
            for dim in shape:
                size *= dim
        else:
            size = shape
        return MockArray([1] * size, shape)
    
    @staticmethod
    def mean(arr, axis=None):
        if hasattr(arr, 'mean'):
            return arr.mean(axis)
        return sum(arr) / len(arr) if arr else 0
    
    @staticmethod
    def std(arr, axis=None):
        if hasattr(arr, 'std'):
            return arr.std(axis)
        if not arr:
            return 0
        mean_val = sum(arr) / len(arr)
        variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
        return variance ** 0.5
    
    @staticmethod
    def concatenate(arrays, axis=0):
        result = []
        for arr in arrays:
            if hasattr(arr, 'data'):
                result.extend(arr.data)
            else:
                result.extend(arr)
        return MockArray(result)
    
    @staticmethod
    def diff(arr, n=1):
        """Calculate differences along axis"""
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
        
        # Apply n-th order differences
        for _ in range(n):
            if len(data) <= 1:
                return MockArray([])
            data = [data[i+1] - data[i] for i in range(len(data)-1)]
        
        return MockArray(data)
    
    @staticmethod
    def where(condition, x, y):
        """Conditional selection from arrays"""
        # Handle different types of condition
        if isinstance(condition, bool):
            # Single boolean condition
            return x if condition else y
        elif hasattr(condition, 'data'):
            # Array-like condition
            cond_data = condition.data
        else:
            # List or other iterable
            cond_data = condition
        
        # Handle x and y values
        if hasattr(x, 'data'):
            x_data = x.data
        elif not isinstance(x, list):
            x_data = [x] * len(cond_data)
        else:
            x_data = x
            
        if hasattr(y, 'data'):
            y_data = y.data
        elif not isinstance(y, list):
            y_data = [y] * len(cond_data)
        else:
            y_data = y
        
        # Apply condition
        result = []
        for i in range(len(cond_data)):
            c = cond_data[i] if i < len(cond_data) else False
            x_val = x_data[i] if i < len(x_data) else (x if not isinstance(x, list) else 0)
            y_val = y_data[i] if i < len(y_data) else (y if not isinstance(y, list) else 0)
            result.append(x_val if c else y_val)
        
        return MockArray(result)
    
    @staticmethod
    def any(arr, axis=None):
        """Check if any element is True"""
        if hasattr(arr, 'data'):
            return any(arr.data) if arr.data else False
        return any(arr) if arr else False
    
    @staticmethod
    def all(arr, axis=None):
        """Check if all elements are True"""
        if hasattr(arr, 'data'):
            return all(arr.data) if arr.data else True
        return all(arr) if arr else True
    
    @staticmethod
    def abs(arr):
        """Absolute values"""
        if hasattr(arr, 'data'):
            return MockArray([abs(x) for x in arr.data])
        return MockArray([abs(x) for x in arr])
    
    @staticmethod
    def min(arr, axis=None):
        """Minimum value"""
        if hasattr(arr, 'data'):
            return min(arr.data) if arr.data else 0
        return min(arr) if arr else 0
    
    @staticmethod
    def max(arr, axis=None):
        """Maximum value"""
        if hasattr(arr, 'data'):
            return max(arr.data) if arr.data else 0
        return max(arr) if arr else 0
    
    @staticmethod
    def sum(arr, axis=None):
        """Sum of values"""
        if hasattr(arr, 'data'):
            return sum(arr.data) if arr.data else 0
        return sum(arr) if arr else 0
    
    @staticmethod
    def full(shape, fill_value):
        """Create array filled with fill_value"""
        if isinstance(shape, (list, tuple)):
            size = 1
            for dim in shape:
                size *= dim
        else:
            size = shape
        return MockArray([fill_value] * size, shape)
    
    @staticmethod
    def arange(start, stop=None, step=1):
        """Create arithmetic sequence"""
        if stop is None:
            stop = start
            start = 0
        
        result = []
        current = start
        while current < stop:
            result.append(current)
            current += step
        return MockArray(result)
    
    @staticmethod
    def average(arr, weights=None):
        """Calculate weighted average"""
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
            
        if weights is None:
            return sum(data) / len(data) if data else 0
        
        if hasattr(weights, 'data'):
            weights = weights.data
            
        if len(data) != len(weights):
            return sum(data) / len(data) if data else 0
            
        weighted_sum = sum(x * w for x, w in zip(data, weights))
        weight_sum = sum(weights)
        return weighted_sum / weight_sum if weight_sum != 0 else 0
    
    @staticmethod
    def cumsum(arr, axis=None):
        """Calculate cumulative sum"""
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
        
        result = []
        cumulative = 0
        for x in data:
            cumulative += x
            result.append(cumulative)
        
        return MockArray(result)
    
    @staticmethod
    def roll(arr, shift, axis=None):
        """Roll array elements along axis"""
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
        
        if len(data) == 0:
            return MockArray([])
        
        # Handle negative shifts
        shift = shift % len(data)
        
        # Roll the data
        rolled_data = data[-shift:] + data[:-shift]
        return MockArray(rolled_data)
    
    @staticmethod
    def maximum(arr1, arr2):
        """Element-wise maximum of two arrays"""
        if hasattr(arr1, 'data'):
            data1 = arr1.data
        else:
            data1 = arr1 if isinstance(arr1, list) else [arr1]
            
        if hasattr(arr2, 'data'):
            data2 = arr2.data
        else:
            data2 = arr2 if isinstance(arr2, list) else [arr2]
        
        # Handle different lengths
        max_len = max(len(data1), len(data2))
        result = []
        
        for i in range(max_len):
            val1 = data1[i] if i < len(data1) else data1[-1] if data1 else 0
            val2 = data2[i] if i < len(data2) else data2[-1] if data2 else 0
            result.append(max(val1, val2))
        
        return MockArray(result)
    
    @staticmethod
    def minimum(arr1, arr2):
        """Element-wise minimum of two arrays"""
        if hasattr(arr1, 'data'):
            data1 = arr1.data
        else:
            data1 = arr1 if isinstance(arr1, list) else [arr1]
            
        if hasattr(arr2, 'data'):
            data2 = arr2.data
        else:
            data2 = arr2 if isinstance(arr2, list) else [arr2]
        
        # Handle different lengths
        max_len = max(len(data1), len(data2))
        result = []
        
        for i in range(max_len):
            val1 = data1[i] if i < len(data1) else data1[-1] if data1 else 0
            val2 = data2[i] if i < len(data2) else data2[-1] if data2 else 0
            result.append(min(val1, val2))
        
        return MockArray(result)
    
    @staticmethod
    def argmax(arr, axis=None):
        """Return indices of maximum values"""
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
        
        if not data:
            return 0
        
        max_val = max(data)
        return data.index(max_val)
    
    @staticmethod
    def argmin(arr, axis=None):
        """Return indices of minimum values"""
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
        
        if not data:
            return 0
        
        min_val = min(data)
        return data.index(min_val)
    
    @staticmethod
    def log(arr):
        """Natural logarithm"""
        import math
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr if isinstance(arr, list) else [arr]
        
        result = []
        for x in data:
            try:
                if x > 0:
                    result.append(math.log(x))
                else:
                    result.append(float('nan'))
            except:
                result.append(float('nan'))
        
        return MockArray(result)
    
    @staticmethod
    def exp(arr):
        """Exponential function"""
        import math
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr if isinstance(arr, list) else [arr]
        
        result = []
        for x in data:
            try:
                result.append(math.exp(x))
            except:
                result.append(float('nan'))
        
        return MockArray(result)
    
    @staticmethod
    def sqrt(arr):
        """Square root"""
        import math
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr if isinstance(arr, list) else [arr]
        
        result = []
        for x in data:
            try:
                if x >= 0:
                    result.append(math.sqrt(x))
                else:
                    result.append(float('nan'))
            except:
                result.append(float('nan'))
        
        return MockArray(result)
    
    @staticmethod
    def isnan(arr):
        """Check for NaN values"""
        import math
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr if isinstance(arr, list) else [arr]
        
        result = []
        for x in data:
            try:
                result.append(math.isnan(x) if isinstance(x, float) else False)
            except:
                result.append(False)
        
        return MockArray(result)
    
    @staticmethod
    def isfinite(arr):
        """Check for finite values"""
        import math
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr if isinstance(arr, list) else [arr]
        
        result = []
        for x in data:
            try:
                result.append(math.isfinite(x) if isinstance(x, (int, float)) else True)
            except:
                result.append(True)
        
        return MockArray(result)
    
    @staticmethod
    def isinf(arr):
        """Check for infinite values"""
        import math
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr if isinstance(arr, list) else [arr]
        
        result = []
        for x in data:
            try:
                result.append(math.isinf(x) if isinstance(x, float) else False)
            except:
                result.append(False)
        
        return MockArray(result)
    
    # Add constants
    nan = float('nan')
    pi = 3.14159265359
    e = 2.71828182846

class MockPandas:
    class Index:
        def __init__(self, data):
            self.data = data
        
        def min(self):
            import datetime
            return datetime.date(2020, 1, 1)
        
        def max(self):
            import datetime
            return datetime.date(2020, 12, 31)
        
        def intersection(self, other):
            return self
        
        def copy(self):
            return MockPandas.Index(self.data.copy() if hasattr(self.data, 'copy') else list(self.data))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, key):
            return self.data[key]

    class DataFrame:
        def __init__(self, data=None, index=None, columns=None):
            if isinstance(data, dict):
                self.data = data
                self.columns = list(data.keys()) if data else []
                index_data = index or list(range(len(next(iter(data.values())) if data else [])))
                self.index = MockPandas.Index(index_data)
            else:
                self.data = {}
                self.columns = columns or []
                self.index = MockPandas.Index(index or [])
            self._shape = (len(self.index), len(self.columns))
        
        @property
        def shape(self):
            return self._shape
        
        @property  
        def empty(self):
            return len(self.data) == 0 and len(self.index) == 0
        
        def __getitem__(self, key):
            if isinstance(key, list):
                # Handle selection of multiple columns
                result_data = {col: self.data.get(col, [0] * len(self.index)) for col in key}
                return MockPandas.DataFrame(result_data, index=self.index, columns=key)
            elif key in self.data:
                return MockPandas.Series(self.data[key])
            return MockPandas.Series([0] * len(self.index))
        
        def __setitem__(self, key, value):
            self.data[key] = value
            if key not in self.columns:
                self.columns.append(key)
        
        def __len__(self):
            return len(self.index)
        
        def isnull(self):
            # Return a DataFrame with all False (no nulls)
            false_data = {col: [False] * len(self.index) for col in self.columns}
            return MockPandas.DataFrame(false_data, index=self.index.data, columns=self.columns)
        
        def any(self):
            # Return a Series with all False
            return MockPandas.Series([False] * len(self.columns))
        
        def __lt__(self, other):
            # Return a DataFrame with all False (no values less than other)
            false_data = {col: [False] * len(self.index) for col in self.columns}
            return MockPandas.DataFrame(false_data, index=self.index.data, columns=self.columns)
        
        def copy(self):
            # Return a copy of the DataFrame
            copied_data = self.data.copy() if hasattr(self.data, 'copy') else dict(self.data)
            copied_index = self.index.copy() if hasattr(self.index, 'copy') else self.index
            copied_columns = self.columns.copy() if hasattr(self.columns, 'copy') else list(self.columns)
            return MockPandas.DataFrame(copied_data, index=copied_index.data if hasattr(copied_index, 'data') else copied_index, columns=copied_columns)
        
        def fillna(self, **kwargs):
            return self
        
        def dropna(self):
            return self
    
    class Series:
        def __init__(self, data=None):
            self.data = data or []
        
        def rolling(self, window):
            """Return a rolling object that supports aggregation functions"""
            class RollingMock:
                def __init__(self, data, window):
                    self.data = data
                    self.window = window
                
                def mean(self):
                    """Calculate rolling mean"""
                    if len(self.data) < self.window:
                        # Return NaN for insufficient data
                        result = [float('nan')] * len(self.data)
                    else:
                        result = []
                        for i in range(len(self.data)):
                            if i < self.window - 1:
                                result.append(float('nan'))
                            else:
                                window_data = self.data[i-self.window+1:i+1]
                                result.append(sum(window_data) / len(window_data))
                    return MockPandas.Series(result)
                
                def std(self):
                    """Calculate rolling standard deviation"""
                    if len(self.data) < self.window:
                        result = [float('nan')] * len(self.data)
                    else:
                        result = []
                        for i in range(len(self.data)):
                            if i < self.window - 1:
                                result.append(float('nan'))
                            else:
                                window_data = self.data[i-self.window+1:i+1]
                                mean_val = sum(window_data) / len(window_data)
                                variance = sum((x - mean_val) ** 2 for x in window_data) / len(window_data)
                                result.append(variance ** 0.5)
                    return MockPandas.Series(result)
                
                def min(self):
                    """Calculate rolling minimum"""
                    if len(self.data) < self.window:
                        result = [float('nan')] * len(self.data)
                    else:
                        result = []
                        for i in range(len(self.data)):
                            if i < self.window - 1:
                                result.append(float('nan'))
                            else:
                                window_data = self.data[i-self.window+1:i+1]
                                result.append(min(window_data))
                    return MockPandas.Series(result)
                
                def max(self):
                    """Calculate rolling maximum"""
                    if len(self.data) < self.window:
                        result = [float('nan')] * len(self.data)
                    else:
                        result = []
                        for i in range(len(self.data)):
                            if i < self.window - 1:
                                result.append(float('nan'))
                            else:
                                window_data = self.data[i-self.window+1:i+1]
                                result.append(max(window_data))
                    return MockPandas.Series(result)
                
                def sum(self):
                    """Calculate rolling sum"""
                    if len(self.data) < self.window:
                        result = [float('nan')] * len(self.data)
                    else:
                        result = []
                        for i in range(len(self.data)):
                            if i < self.window - 1:
                                result.append(float('nan'))
                            else:
                                window_data = self.data[i-self.window+1:i+1]
                                result.append(sum(window_data))
                    return MockPandas.Series(result)
                
                def apply(self, func, raw=False):
                    """Apply custom function to rolling windows"""
                    if len(self.data) < self.window:
                        result = [float('nan')] * len(self.data)
                    else:
                        result = []
                        for i in range(len(self.data)):
                            if i < self.window - 1:
                                result.append(float('nan'))
                            else:
                                window_data = self.data[i-self.window+1:i+1]
                                try:
                                    result.append(func(window_data))
                                except:
                                    result.append(float('nan'))
                    return MockPandas.Series(result)
            
            return RollingMock(self.data, window)
        
        def mean(self):
            if not self.data:
                return 0
            return sum(self.data) / len(self.data)
        
        def std(self):
            if not self.data:
                return 0
            mean_val = self.mean()
            variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
            return variance ** 0.5
        
        def min(self):
            return min(self.data) if self.data else 0
        
        def max(self):
            return max(self.data) if self.data else 0
        
        def sum(self):
            return sum(self.data) if self.data else 0
        
        def any(self):
            return any(self.data) if self.data else False
        
        def apply(self, func):
            """Apply function - return a simple result"""
            try:
                if self.data:
                    result = func(self.data)
                    return MockPandas.Series([result] * len(self.data))
                return MockPandas.Series([0])
            except:
                return MockPandas.Series([0] * len(self.data))
        
        def __eq__(self, other):
            # Return a Series of boolean values
            if isinstance(other, (int, float)):
                return MockPandas.Series([x == other for x in self.data])
            return MockPandas.Series([False] * len(self.data))
        
        def __gt__(self, other):
            # Return a Series of boolean values
            if isinstance(other, (int, float)):
                return MockPandas.Series([x > other for x in self.data])
            return MockPandas.Series([False] * len(self.data))
        
        def __lt__(self, other):
            # Return a Series of boolean values
            if isinstance(other, (int, float)):
                return MockPandas.Series([x < other for x in self.data])
            return MockPandas.Series([False] * len(self.data))
        
        def ewm(self, span=None, alpha=None):
            """Exponentially weighted operations"""
            class EWMMock:
                def __init__(self, data, span, alpha):
                    self.data = data
                    self.span = span
                    self.alpha = alpha
                
                def mean(self):
                    """Calculate exponentially weighted mean"""
                    if not self.data:
                        return MockPandas.Series([])
                    
                    # Simple exponential moving average implementation
                    alpha = self.alpha if self.alpha else 2.0 / (self.span + 1) if self.span else 0.1
                    result = [self.data[0]]  # First value stays the same
                    
                    for i in range(1, len(self.data)):
                        ema_val = alpha * self.data[i] + (1 - alpha) * result[i-1]
                        result.append(ema_val)
                    
                    return MockPandas.Series(result)
            
            return EWMMock(self.data, span, alpha)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, key):
            if isinstance(key, slice):
                return MockPandas.Series(self.data[key])
            return self.data[key]
        
        # Arithmetic operations
        def __add__(self, other):
            if isinstance(other, (int, float)):
                return MockPandas.Series([x + other for x in self.data])
            elif hasattr(other, 'data'):
                return MockPandas.Series([x + y for x, y in zip(self.data, other.data)])
            return MockPandas.Series([x + other for x in self.data])
        
        def __radd__(self, other):
            return self.__add__(other)
        
        def __sub__(self, other):
            if isinstance(other, (int, float)):
                return MockPandas.Series([x - other for x in self.data])
            elif hasattr(other, 'data'):
                return MockPandas.Series([x - y for x, y in zip(self.data, other.data)])
            return MockPandas.Series([x - other for x in self.data])
        
        def __rsub__(self, other):
            if isinstance(other, (int, float)):
                return MockPandas.Series([other - x for x in self.data])
            elif hasattr(other, 'data'):
                return MockPandas.Series([y - x for x, y in zip(self.data, other.data)])
            return MockPandas.Series([other - x for x in self.data])
        
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                return MockPandas.Series([x * other for x in self.data])
            elif hasattr(other, 'data'):
                return MockPandas.Series([x * y for x, y in zip(self.data, other.data)])
            return MockPandas.Series([x * other for x in self.data])
        
        def __rmul__(self, other):
            return self.__mul__(other)
        
        def __truediv__(self, other):
            if isinstance(other, (int, float)):
                return MockPandas.Series([x / other if other != 0 else 0 for x in self.data])
            elif hasattr(other, 'data'):
                return MockPandas.Series([x / y if y != 0 else 0 for x, y in zip(self.data, other.data)])
            return MockPandas.Series([x / other if other != 0 else 0 for x in self.data])
        
        def __rtruediv__(self, other):
            if isinstance(other, (int, float)):
                return MockPandas.Series([other / x if x != 0 else 0 for x in self.data])
            elif hasattr(other, 'data'):
                return MockPandas.Series([y / x if x != 0 else 0 for x, y in zip(self.data, other.data)])
            return MockPandas.Series([other / x if x != 0 else 0 for x in self.data])
        
        def isna(self):
            """Check for NaN values"""
            import math
            return MockPandas.Series([math.isnan(x) if isinstance(x, float) else False for x in self.data])
        
        def isnull(self):
            """Alias for isna"""
            return self.isna()
        
        def all(self):
            """Check if all values are True"""
            return all(self.data) if self.data else True
        
        @property
        def values(self):
            return MockArray(self.data)

class MockStreamlit:
    @staticmethod
    def set_page_config(**kwargs):
        pass
    
    @staticmethod
    def title(text):
        print(f"TITLE: {text}")
    
    @staticmethod
    def markdown(text):
        print(f"MARKDOWN: {text}")
    
    @staticmethod
    def header(text):
        print(f"HEADER: {text}")
    
    @staticmethod
    def subheader(text):
        print(f"SUBHEADER: {text}")
    
    @staticmethod
    def write(text):
        print(f"WRITE: {text}")
    
    @staticmethod
    def button(text, **kwargs):
        return False
    
    @staticmethod
    def selectbox(text, options, **kwargs):
        return options[0] if options else None
    
    @staticmethod
    def columns(n):
        class MockColumn:
            def __enter__(self):
                return MockStreamlit
            
            def __exit__(self, *args):
                pass
        
        return [MockColumn() for _ in range(n)]
    
    class sidebar:
        @staticmethod
        def title(text):
            print(f"SIDEBAR TITLE: {text}")
        
        @staticmethod
        def selectbox(text, options, **kwargs):
            print(f"SIDEBAR SELECTBOX: {text}")
            return options[0] if options else None
    
    @staticmethod
    def error(text):
        print(f"ERROR: {text}")
    
    @staticmethod
    def warning(text):
        print(f"WARNING: {text}")
    
    @staticmethod
    def success(text):
        print(f"SUCCESS: {text}")

class MockPlotly:
    class graph_objects:
        class Figure:
            def __init__(self, *args, **kwargs):
                pass
            
            def add_trace(self, *args, **kwargs):
                pass
            
            def update_layout(self, *args, **kwargs):
                pass
            
            def show(self):
                print("Plotly figure would be displayed here")
        
        Bar = Figure
        Scatter = Figure
        
    class express:
        @staticmethod
        def line(*args, **kwargs):
            return MockPlotly.graph_objects.Figure()
        
        @staticmethod
        def bar(*args, **kwargs):
            return MockPlotly.graph_objects.Figure()
    
    class subplots:
        @staticmethod
        def make_subplots(*args, **kwargs):
            return MockPlotly.graph_objects.Figure()

class MockSklearn:
    class model_selection:
        @staticmethod
        def train_test_split(*arrays, **kwargs):
            # Return mock splits
            return [arr[:len(arr)//2] for arr in arrays] + [arr[len(arr)//2:] for arr in arrays]
    
    class preprocessing:
        class StandardScaler:
            def fit(self, X):
                return self
            
            def transform(self, X):
                return X
            
            def fit_transform(self, X):
                return X
            
            def inverse_transform(self, X):
                return X
        
        class MinMaxScaler:
            def fit(self, X):
                return self
            
            def transform(self, X):
                return X
            
            def fit_transform(self, X):
                return X
            
            def inverse_transform(self, X):
                return X
    
    class metrics:
        @staticmethod
        def accuracy_score(*args, **kwargs):
            return 0.85
        
        @staticmethod
        def precision_score(*args, **kwargs):
            return 0.83
        
        @staticmethod
        def recall_score(*args, **kwargs):
            return 0.87
        
        @staticmethod
        def f1_score(*args, **kwargs):
            return 0.85
        
        @staticmethod
        def roc_auc_score(*args, **kwargs):
            return 0.89
        
        @staticmethod
        def average_precision_score(*args, **kwargs):
            return 0.88
        
        @staticmethod
        def matthews_corrcoef(*args, **kwargs):
            return 0.75
        
        @staticmethod
        def log_loss(*args, **kwargs):
            return 0.25
        
        @staticmethod
        def confusion_matrix(*args, **kwargs):
            return MockArray([[50, 10], [12, 48]])
        
        @staticmethod
        def classification_report(*args, **kwargs):
            return "Mock classification report"
        
        @staticmethod
        def roc_curve(*args, **kwargs):
            return MockArray([0, 0.5, 1]), MockArray([0, 0.5, 1]), MockArray([0.5, 0.7, 0.9])
        
        @staticmethod
        def precision_recall_curve(*args, **kwargs):
            return MockArray([1, 0.8, 0.6]), MockArray([0, 0.5, 1]), MockArray([0.5, 0.7, 0.9])

class MockYfinance:
    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
        
        def history(self, **kwargs):
            # Return mock stock data
            import datetime
            dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
            data = {
                'Open': [100 + i * 0.1 for i in range(100)],
                'High': [101 + i * 0.1 for i in range(100)],
                'Low': [99 + i * 0.1 for i in range(100)],
                'Close': [100.5 + i * 0.1 for i in range(100)],
                'Volume': [1000000 + i * 1000 for i in range(100)]
            }
            df = MockPandas.DataFrame(data, index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            df.data = data  # Ensure data is accessible
            return df
    
    @staticmethod
    def download(*args, **kwargs):
        # Return mock data for multiple symbols
        data = {'Close': [100 + i * 0.1 for i in range(100)]}
        return MockPandas.DataFrame(data, columns=['Close'])

# Install mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['pandas'] = MockPandas()
sys.modules['streamlit'] = MockStreamlit()
sys.modules['plotly'] = MockPlotly()
sys.modules['plotly.graph_objects'] = MockPlotly.graph_objects()
sys.modules['plotly.express'] = MockPlotly.express()
sys.modules['plotly.subplots'] = MockPlotly.subplots()
sys.modules['sklearn'] = MockSklearn()
sys.modules['sklearn.model_selection'] = MockSklearn.model_selection()
sys.modules['sklearn.preprocessing'] = MockSklearn.preprocessing()
sys.modules['sklearn.metrics'] = MockSklearn.metrics()
sys.modules['yfinance'] = MockYfinance()

class MockTensorflow:
    class keras:
        class layers:
            class Layer:
                def __init__(self, **kwargs):
                    pass
                
                def build(self, input_shape):
                    pass
                
                def call(self, inputs):
                    return inputs
            
            class Dense:
                def __init__(self, units, activation=None, **kwargs):
                    self.units = units
                    self.activation = activation
                
                def __call__(self, inputs):
                    return inputs
            
            class Dropout:
                def __init__(self, rate, **kwargs):
                    self.rate = rate
                
                def __call__(self, inputs):
                    return inputs
            
            class RNN:
                def __init__(self, cell, **kwargs):
                    self.cell = cell
                
                def __call__(self, inputs):
                    return inputs
        
        class Model:
            def __init__(self, *args, **kwargs):
                pass
            
            def compile(self, *args, **kwargs):
                pass
            
            def fit(self, *args, **kwargs):
                return {'loss': [0.1], 'accuracy': [0.5]}
            
            def predict(self, X):
                if hasattr(X, 'shape'):
                    return MockArray([0.5] * X.shape[0])
                return MockArray([0.5] * len(X))
            
            def evaluate(self, X, y):
                return {'loss': 0.1, 'accuracy': 0.5}
        
        class callbacks:
            class EarlyStopping:
                def __init__(self, **kwargs):
                    pass
            
            class ReduceLROnPlateau:
                def __init__(self, **kwargs):
                    pass
        
        @staticmethod
        def Input(shape):
            return None
    
    class nn:
        @staticmethod
        def tanh(x):
            return x
        
        @staticmethod
        def softmax(x, axis=None):
            return x
    
    @staticmethod
    def tensordot(a, b, axes):
        return a
    
    @staticmethod
    def reduce_sum(x, axis=None):
        return x

sys.modules['tensorflow'] = MockTensorflow()
sys.modules['tensorflow.keras'] = MockTensorflow.keras()
sys.modules['tensorflow.keras.layers'] = MockTensorflow.keras.layers()
sys.modules['tensorflow.keras.callbacks'] = MockTensorflow.keras.callbacks()

# Add numpy shortcuts
np = MockNumpy()
sys.modules['numpy'].np = np

# Add pandas shortcuts  
pd = MockPandas()
sys.modules['pandas'].pd = pd

# Add streamlit shortcuts
st = MockStreamlit()
sys.modules['streamlit'].st = st

print("Mock dependencies installed successfully")