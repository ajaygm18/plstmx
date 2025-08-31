"""
Mock implementations for missing dependencies
This allows the application to run without installing packages
"""

import sys
import os
from types import ModuleType

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
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def flatten(self):
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

class MockNumpy:
    array = MockArray
    ndarray = MockArray
    
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
    def diff(arr):
        if hasattr(arr, 'data'):
            data = arr.data
        else:
            data = arr
        return MockArray([data[i+1] - data[i] for i in range(len(data)-1)])
    
    @staticmethod
    def where(condition, x, y):
        # Simple mock implementation
        return MockArray([x if c else y for c in condition])

class MockPandas:
    class DataFrame:
        def __init__(self, data=None, index=None, columns=None):
            self.data = data or {}
            self.index = index or []
            self.columns = columns or []
            self._shape = (len(self.index), len(self.columns))
        
        @property
        def shape(self):
            return self._shape
        
        def __getitem__(self, key):
            if key in self.data:
                return self.data[key]
            return MockPandas.Series([0] * len(self.index))
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def fillna(self, **kwargs):
            return self
        
        def dropna(self):
            return self
    
    class Series:
        def __init__(self, data=None):
            self.data = data or []
        
        def rolling(self, window):
            return self
        
        def mean(self):
            return MockPandas.Series([0] * len(self.data))
        
        def ewm(self, span):
            return self
        
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
            df = MockPandas.DataFrame(data, index=dates)
            return df
    
    @staticmethod
    def download(*args, **kwargs):
        # Return mock data for multiple symbols
        return MockPandas.DataFrame({
            'Close': [100 + i * 0.1 for i in range(100)]
        })

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