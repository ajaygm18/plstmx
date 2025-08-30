"""
Bayesian Optimization implementation for hyperparameter tuning
As described in PMC10963254 research paper
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei
import logging
from typing import Dict, List, Tuple, Callable, Any
from functools import partial

from config.settings import BAYESIAN_OPT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning of PLSTM-TAL model
    """
    
    def __init__(self, n_calls: int = None, n_initial_points: int = None, random_state: int = None):
        """
        Initialize Bayesian Optimizer
        
        Args:
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            random_state: Random state for reproducibility
        """
        self.n_calls = n_calls or BAYESIAN_OPT_CONFIG['n_calls']
        self.n_initial_points = n_initial_points or BAYESIAN_OPT_CONFIG['n_initial_points']
        self.random_state = random_state or BAYESIAN_OPT_CONFIG['random_state']
        
        # Define search space for PLSTM-TAL hyperparameters
        self.search_space = [
            Integer(32, 256, name='lstm_units'),           # LSTM units
            Integer(16, 128, name='attention_units'),      # Attention units
            Real(0.1, 0.5, name='dropout_rate'),          # Dropout rate
            Real(1e-5, 1e-2, name='learning_rate'),       # Learning rate
            Integer(16, 64, name='batch_size'),            # Batch size
            Integer(30, 120, name='sequence_length'),      # Sequence length
        ]
        
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
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
                
                # Train model with given parameters
                model, history = model_trainer(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    **params
                )
                
                # Evaluate on validation set
                val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
                
                # Calculate F1 score
                val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
                
                # Use negative accuracy as objective (since we minimize)
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
        Perform Bayesian optimization
        
        Args:
            objective_function: Function to optimize
            
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Starting Bayesian optimization with {self.n_calls} calls")
            
            # Perform optimization
            result = gp_minimize(
                func=objective_function,
                dimensions=self.search_space,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                random_state=self.random_state,
                acq_func='EI',  # Expected Improvement
                n_jobs=1
            )
            
            # Extract best parameters
            best_params_list = result.x
            self.best_params = {
                'lstm_units': best_params_list[0],
                'attention_units': best_params_list[1],
                'dropout_rate': best_params_list[2],
                'learning_rate': best_params_list[3],
                'batch_size': best_params_list[4],
                'sequence_length': best_params_list[5]
            }
            
            self.best_score = -result.fun  # Convert back to positive
            
            optimization_results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_history': self.optimization_history,
                'convergence_trace': result.func_vals,
                'n_calls': len(result.func_vals)
            }
            
            logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
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
    Optimize PLSTM-TAL hyperparameters using Bayesian optimization
    
    Args:
        model_trainer: Function to train PLSTM-TAL model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_calls: Number of optimization calls
        
    Returns:
        Tuple of (best_params, optimization_results)
    """
    try:
        logger.info("Starting hyperparameter optimization for PLSTM-TAL")
        
        # Initialize optimizer
        optimizer = BayesianOptimizer(n_calls=n_calls)
        
        # Define objective function
        objective = optimizer.define_objective_function(
            model_trainer, X_train, y_train, X_val, y_val
        )
        
        # Perform optimization
        results = optimizer.optimize(objective)
        
        # Get best parameters
        best_params = optimizer.get_best_params()
        
        # Analyze parameter importance
        param_importance = optimizer.analyze_parameter_importance()
        results['parameter_importance'] = param_importance
        
        logger.info("Hyperparameter optimization completed successfully")
        
        return best_params, results
        
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
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
