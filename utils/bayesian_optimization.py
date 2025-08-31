"""
Bayesian Optimization implementation for hyperparameter tuning
As described in PMC10963254 research paper
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Callable, Any
from functools import partial

# Try to import scikit-optimize, use fallback if not available
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    logging.warning("scikit-optimize not available, using fallback optimization")
    SKOPT_AVAILABLE = False
    
    # Fallback implementations
    class Real:
        def __init__(self, low, high, name=None):
            self.low = low
            self.high = high
            self.name = name
    
    class Integer:
        def __init__(self, low, high, name=None):
            self.low = low
            self.high = high
            self.name = name
    
    def use_named_args(dimensions):
        def decorator(func):
            def wrapper(params):
                # Convert list of params to dict using dimension names
                param_dict = {}
                for i, dim in enumerate(dimensions):
                    param_dict[dim.name] = params[i]
                return func(**param_dict)
            return wrapper
        return decorator
    
    def gp_minimize(func, dimensions, n_calls=50, n_initial_points=10, random_state=42):
        # Fallback: random search
        np.random.seed(random_state)
        best_score = float('inf')
        best_params = None
        results = []
        
        for _ in range(n_calls):
            params = []
            for dim in dimensions:
                if isinstance(dim, Real):
                    param = np.random.uniform(dim.low, dim.high)
                else:  # Integer
                    param = np.random.randint(dim.low, dim.high + 1)
                params.append(param)
            
            score = func(params)
            results.append(score)
            
            if score < best_score:
                best_score = score
                best_params = params
        
        # Create a mock result object
        class MockResult:
            def __init__(self, x, fun, func_vals):
                self.x = x
                self.fun = fun
                self.func_vals = func_vals
        
        return MockResult(best_params, best_score, results)

from config.settings import BAYESIAN_OPT_CONFIG, EXTENDED_TRAINING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning of PLSTM-TAL model
    """
    
    def __init__(self, n_calls: int = None, n_initial_points: int = None, random_state: int = None):
        """
        Initialize Bayesian Optimizer with enhanced configuration for PMC10963254
        
        Args:
            n_calls: Number of optimization calls (increased for extended optimization)
            n_initial_points: Number of initial random points
            random_state: Random state for reproducibility
        """
        self.n_calls = n_calls or BAYESIAN_OPT_CONFIG['n_calls']
        self.n_initial_points = n_initial_points or BAYESIAN_OPT_CONFIG['n_initial_points']
        self.random_state = random_state or BAYESIAN_OPT_CONFIG['random_state']
        
        # Enhanced search space for PLSTM-TAL hyperparameters (PMC10963254 compliant)
        self.search_space = [
            Integer(64, 512, name='lstm_units'),           # Expanded LSTM units range
            Integer(32, 256, name='attention_units'),      # Expanded attention units range
            Real(0.05, 0.6, name='dropout_rate'),          # Wider dropout rate range
            Real(1e-6, 1e-1, name='learning_rate'),        # Expanded learning rate range
            Integer(16, 128, name='batch_size'),           # Expanded batch size range
            Integer(30, 120, name='sequence_length'),      # Sequence length range
            Integer(200, 500, name='epochs'),              # Extended epochs range for long training
            Real(0.1, 0.9, name='recurrent_dropout'),      # Recurrent dropout for LSTM
            Integer(1, 4, name='lstm_layers'),             # Number of LSTM layers
            Real(0.0001, 0.1, name='l1_reg'),             # L1 regularization
            Real(0.0001, 0.1, name='l2_reg'),             # L2 regularization
        ]
        
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        self.target_accuracy = EXTENDED_TRAINING_CONFIG['target_accuracy']  # 70% target
        
    def define_objective_function(self, model_trainer: Callable, X_train: np.ndarray, 
                                y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Callable:
        """
        Define the objective function for optimization
        
        Args:
            model_trainer: Function to train model with given parameters
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Objective function for optimization
        """
        @use_named_args(self.search_space)
        def objective(**params):
            try:
                logger.info(f"Evaluating hyperparameters: {params}")
                
                # Enhanced training with extended configuration
                start_time = time.time()
                
                # Train model with given parameters and extended configuration
                model, history = model_trainer(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    **params
                )
                
                training_time = time.time() - start_time
                
                # Evaluate on validation set with comprehensive metrics
                val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
                
                # Calculate F1 score and other metrics
                val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
                
                # Enhanced objective function with multiple criteria
                # Primary: maximize accuracy, Secondary: minimize loss, Tertiary: maximize F1
                accuracy_weight = 0.7
                f1_weight = 0.2
                loss_weight = 0.1
                
                # Composite objective (negative because we minimize)
                objective_value = -(accuracy_weight * val_accuracy + f1_weight * val_f1 - loss_weight * val_loss)
                
                # Store enhanced evaluation results
                eval_results = {
                    'params': params,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'objective_value': objective_value,
                    'training_time': training_time,
                    'target_met': val_accuracy >= self.target_accuracy
                }
                
                self.optimization_history.append(eval_results)
                
                logger.info(f"Validation metrics - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, "
                           f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                           f"Training time: {training_time:.2f}s")
                
                # Check if target accuracy is reached
                if val_accuracy >= self.target_accuracy:
                    logger.info(f"🎯 Target accuracy {self.target_accuracy:.1%} achieved! "
                               f"Current accuracy: {val_accuracy:.4f}")
                
                return objective_value
                objective_value = -val_accuracy
                
                # Store evaluation results
                eval_results = {
                    'params': params,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                    'objective_value': objective_value
                }
                
                self.optimization_history.append(eval_results)
                
                logger.info(f"Validation accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
                
                return objective_value
                
            except Exception as e:
                logger.error(f"Error in objective function: {str(e)}")
                return 1.0  # Return large positive value for failed evaluations
        
        return objective
    
    def optimize(self, objective_function: Callable) -> Dict[str, Any]:
        """
        Perform enhanced Bayesian optimization for extended training
        
        Args:
            objective_function: Function to optimize
            
        Returns:
            Optimization results with extended metrics
        """
        try:
            logger.info(f"Starting extended Bayesian optimization with {self.n_calls} calls")
            logger.info(f"Target accuracy: {self.target_accuracy:.1%}")
            
            # Enhanced optimization configuration
            optimization_config = {
                'func': objective_function,
                'dimensions': self.search_space,
                'n_calls': self.n_calls,
                'n_initial_points': self.n_initial_points,
                'random_state': self.random_state,
                'acq_func': BAYESIAN_OPT_CONFIG.get('acq_func', 'EI'),
                'n_jobs': BAYESIAN_OPT_CONFIG.get('n_jobs', 1),
                'verbose': BAYESIAN_OPT_CONFIG.get('verbose', True)
            }
            
            start_time = time.time()
            
            # Perform optimization with no timeout
            result = gp_minimize(**optimization_config)
            
            optimization_time = time.time() - start_time
            
            # Extract best parameters with enhanced parameter mapping
            best_params_list = result.x
            self.best_params = {
                'lstm_units': int(best_params_list[0]),
                'attention_units': int(best_params_list[1]),
                'dropout_rate': float(best_params_list[2]),
                'learning_rate': float(best_params_list[3]),
                'batch_size': int(best_params_list[4]),
                'sequence_length': int(best_params_list[5]),
                'epochs': int(best_params_list[6]),
                'recurrent_dropout': float(best_params_list[7]),
                'lstm_layers': int(best_params_list[8]),
                'l1_reg': float(best_params_list[9]),
                'l2_reg': float(best_params_list[10])
            }
            
            self.best_score = -result.fun  # Convert back to positive
            
            # Enhanced optimization results
            optimization_results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_history': self.optimization_history,
                'convergence_trace': result.func_vals,
                'n_calls': len(result.func_vals),
                'optimization_time': optimization_time,
                'target_accuracy': self.target_accuracy,
                'target_achieved': self.best_score >= self.target_accuracy,
                'best_accuracy_achieved': max([r['val_accuracy'] for r in self.optimization_history]) if self.optimization_history else 0,
                'convergence_info': {
                    'converged': len(result.func_vals) < self.n_calls,
                    'improvement_ratio': (max(result.func_vals) - min(result.func_vals)) / abs(max(result.func_vals)) if result.func_vals else 0
                }
            }
            
            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            logger.info(f"Best accuracy: {optimization_results['best_accuracy_achieved']:.4f}")
            logger.info(f"Target achieved: {optimization_results['target_achieved']}")
            logger.info(f"Best parameters: {self.best_params}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in enhanced Bayesian optimization: {str(e)}")
            raise
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found
        
        Returns:
            Best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("Optimization has not been performed yet")
        
        return self.best_params
    
    def plot_convergence(self, save_path: str = None) -> None:
        """
        Plot optimization convergence
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.optimization_history:
                logger.warning("No optimization history available for plotting")
                return
            
            # Extract scores
            scores = [result['val_accuracy'] for result in self.optimization_history]
            
            # Plot convergence
            plt.figure(figsize=(10, 6))
            plt.plot(scores, 'b-', linewidth=2, label='Validation Accuracy')
            plt.axhline(y=max(scores), color='r', linestyle='--', label=f'Best Score: {max(scores):.4f}')
            plt.xlabel('Iteration')
            plt.ylabel('Validation Accuracy')
            plt.title('Bayesian Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Convergence plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting convergence: {str(e)}")
    
    def analyze_parameter_importance(self) -> Dict[str, float]:
        """
        Analyze the importance of different hyperparameters
        
        Returns:
            Parameter importance scores
        """
        try:
            if len(self.optimization_history) < 10:
                logger.warning("Insufficient optimization history for parameter analysis")
                return {}
            
            # Extract parameter values and scores
            param_names = list(self.search_space)
            param_values = {name: [] for name in [space.name for space in self.search_space]}
            scores = []
            
            for result in self.optimization_history:
                params = result['params']
                for param_name in param_values.keys():
                    param_values[param_name].append(params[param_name])
                scores.append(result['val_accuracy'])
            
            # Calculate correlations (simple importance measure)
            importance = {}
            for param_name, values in param_values.items():
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param_name] = abs(correlation)  # Use absolute correlation
            
            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v/total_importance for k, v in importance.items()}
            
            logger.info("Parameter importance analysis completed")
            return importance
            
        except Exception as e:
            logger.error(f"Error analyzing parameter importance: {str(e)}")
            return {}

def optimize_plstm_tal_hyperparameters(model_trainer: Callable, 
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray,
                                     n_calls: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Enhanced PLSTM-TAL hyperparameter optimization for PMC10963254 compliance
    
    Args:
        model_trainer: Function to train PLSTM-TAL model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_calls: Number of optimization calls (increased for extended optimization)
        
    Returns:
        Tuple of (best_params, comprehensive_optimization_results)
    """
    try:
        logger.info("Starting enhanced hyperparameter optimization for PLSTM-TAL")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Initialize enhanced optimizer
        optimizer = BayesianOptimizer(n_calls=n_calls)
        
        # Define objective function with extended training support
        objective = optimizer.define_objective_function(
            model_trainer, X_train, y_train, X_val, y_val
        )
        
        # Configure for extended training (disable timeouts)
        import tensorflow as tf
        if hasattr(tf.config, 'experimental'):
            # Enable memory growth to handle long training sessions
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        logger.info(f"Starting optimization with {optimizer.n_calls} calls")
        logger.info(f"Target accuracy: {optimizer.target_accuracy:.1%}")
        
        # Perform enhanced optimization
        start_time = time.time()
        results = optimizer.optimize(objective)
        total_time = time.time() - start_time
        
        # Get best parameters
        best_params = optimizer.get_best_params()
        
        # Enhanced analysis
        param_importance = optimizer.analyze_parameter_importance()
        results['parameter_importance'] = param_importance
        results['total_optimization_time'] = total_time
        results['average_trial_time'] = total_time / len(optimizer.optimization_history) if optimizer.optimization_history else 0
        
        # Check if target was achieved
        best_accuracy = max([r['val_accuracy'] for r in optimizer.optimization_history]) if optimizer.optimization_history else 0
        target_achieved = best_accuracy >= optimizer.target_accuracy
        
        logger.info("Enhanced hyperparameter optimization completed successfully")
        logger.info(f"Total optimization time: {total_time:.2f}s")
        logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
        logger.info(f"Target accuracy ({optimizer.target_accuracy:.1%}) achieved: {target_achieved}")
        logger.info(f"Number of trials: {len(optimizer.optimization_history)}")
        
        if target_achieved:
            logger.info("🎯 SUCCESS: Target accuracy achieved!")
        else:
            logger.warning(f"⚠️  Target accuracy not achieved. Best: {best_accuracy:.4f}, Target: {optimizer.target_accuracy:.4f}")
        
        return best_params, results
        
    except Exception as e:
        logger.error(f"Error in enhanced hyperparameter optimization: {str(e)}")
        raise

class GridSearchOptimizer:
    """
    Alternative grid search optimizer for comparison
    """
    
    def __init__(self):
        # Define smaller grid for computational efficiency
        self.param_grid = {
            'lstm_units': [64, 128, 256],
            'attention_units': [32, 64],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'sequence_length': [60, 90]
        }
        
    def grid_search(self, model_trainer: Callable, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Perform grid search optimization
        
        Args:
            model_trainer: Function to train model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuple of (best_params, all_results)
        """
        try:
            logger.info("Starting grid search optimization")
            
            from itertools import product
            
            # Generate all parameter combinations
            param_names = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())
            
            all_results = []
            best_score = -1
            best_params = None
            
            total_combinations = np.prod([len(values) for values in param_values])
            logger.info(f"Total parameter combinations: {total_combinations}")
            
            for i, combination in enumerate(product(*param_values)):
                params = dict(zip(param_names, combination))
                
                logger.info(f"Evaluating combination {i+1}/{total_combinations}: {params}")
                
                try:
                    # Train model
                    model, history = model_trainer(
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        **params
                    )
                    
                    # Evaluate
                    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
                    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
                    
                    result = {
                        'params': params,
                        'val_accuracy': val_accuracy,
                        'val_f1': val_f1,
                        'val_loss': val_loss
                    }
                    
                    all_results.append(result)
                    
                    # Update best
                    if val_accuracy > best_score:
                        best_score = val_accuracy
                        best_params = params.copy()
                    
                    logger.info(f"Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating combination: {str(e)}")
                    continue
            
            logger.info(f"Grid search completed. Best score: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return best_params, all_results
            
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise

if __name__ == "__main__":
    # Test Bayesian optimization
    def dummy_model_trainer(X_train, y_train, X_val, y_val, **params):
        """Dummy trainer for testing"""
        # Simulate model training
        import time
        time.sleep(0.1)  # Simulate training time
        
        # Return dummy model and history
        class DummyModel:
            def evaluate(self, X, y, verbose=0):
                # Return random metrics
                return np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()
        
        return DummyModel(), {}
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 60, 32)
    y_train = np.random.randint(0, 2, (100, 1))
    X_val = np.random.randn(50, 60, 32)
    y_val = np.random.randint(0, 2, (50, 1))
    
    # Test optimization
    best_params, results = optimize_plstm_tal_hyperparameters(
        dummy_model_trainer, X_train, y_train, X_val, y_val, n_calls=5
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {results['best_score']:.4f}")
