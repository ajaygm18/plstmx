"""
Complete training pipeline for PLSTM-TAL model
Orchestrates the entire training process including data preprocessing,
model training, hyperparameter optimization, and benchmark comparison
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, Optional
import os
from datetime import datetime

# Import custom modules
from models.plstm_tal import PLSTMTAL
from models.benchmark_models import BenchmarkModelsRunner
from utils.bayesian_optimization import optimize_plstm_tal_hyperparameters
from utils.evaluation_metrics import EvaluationMetrics
from config.settings import PLSTM_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PLSTMTALTrainer:
    """
    Complete training pipeline for PLSTM-TAL stock market prediction
    """
    
    def __init__(self):
        self.evaluator = EvaluationMetrics()
        self.trained_models = {}
        self.training_history = {}
        self.benchmark_results = {}
        
    def create_plstm_tal_trainer(self, **config_override) -> callable:
        """
        Create a trainer function for PLSTM-TAL with given configuration
        
        Args:
            **config_override: Configuration parameters to override
            
        Returns:
            Trainer function for Bayesian optimization
        """
        def trainer(X_train, y_train, X_val, y_val, **params):
            try:
                # Merge default config with optimization parameters
                model_config = {**PLSTM_CONFIG, **config_override, **params}
                
                # Initialize model
                sequence_length = int(params.get('sequence_length', model_config['sequence_length']))
                n_features = X_train.shape[2]
                
                model = PLSTMTAL(
                    sequence_length=sequence_length,
                    n_features=n_features,
                    lstm_units=int(params.get('lstm_units', model_config['lstm_units'])),
                    attention_units=int(params.get('attention_units', model_config['attention_units'])),
                    dropout_rate=float(params.get('dropout_rate', model_config['dropout_rate']))
                )
                
                # Build model
                model.build_model()
                
                # Prepare data with new sequence length if needed
                if sequence_length != X_train.shape[1]:
                    # Re-create sequences with new length
                    logger.info(f"Re-creating sequences with length {sequence_length}")
                    # This is a simplified approach - in practice, you'd need to recreate from raw features
                    min_seq_len = min(sequence_length, X_train.shape[1])
                    X_train = X_train[:, -min_seq_len:, :]
                    X_val = X_val[:, -min_seq_len:, :]
                
                # Train model
                history = model.train(
                    X_train, y_train,
                    validation_split=0.0,  # We provide explicit validation data
                    epochs=int(params.get('epochs', model_config['epochs'])),
                    batch_size=int(params.get('batch_size', model_config['batch_size'])),
                    verbose=0
                )
                
                return model, history
                
            except Exception as e:
                logger.error(f"Error in PLSTM-TAL trainer: {str(e)}")
                raise
        
        return trainer
    
    def train_plstm_tal(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       use_bayesian_opt: bool = True,
                       **config_override) -> Tuple[PLSTMTAL, Dict, Dict]:
        """
        Train PLSTM-TAL model with optional hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            use_bayesian_opt: Whether to use Bayesian optimization
            **config_override: Configuration parameters to override
            
        Returns:
            Tuple of (trained_model, training_history, best_params)
        """
        try:
            logger.info("Starting PLSTM-TAL training...")
            
            if use_bayesian_opt:
                logger.info("Using Bayesian optimization for hyperparameter tuning")
                
                # Create trainer function
                trainer_func = self.create_plstm_tal_trainer(**config_override)
                
                # Optimize hyperparameters
                best_params, optimization_results = optimize_plstm_tal_hyperparameters(
                    trainer_func, X_train, y_train, X_val, y_val
                )
                
                logger.info(f"Optimization completed. Best parameters: {best_params}")
                
                # Train final model with best parameters
                final_trainer = self.create_plstm_tal_trainer(**config_override)
                model, history = final_trainer(X_train, y_train, X_val, y_val, **best_params)
                
            else:
                logger.info("Training with default parameters")
                
                # Use default parameters
                model_config = {**PLSTM_CONFIG, **config_override}
                best_params = model_config
                
                model = PLSTMTAL(
                    sequence_length=model_config['sequence_length'],
                    n_features=X_train.shape[2],
                    lstm_units=model_config['lstm_units'],
                    attention_units=model_config['attention_units'],
                    dropout_rate=model_config['dropout_rate']
                )
                
                model.build_model()
                
                history = model.train(
                    X_train, y_train,
                    validation_split=0.0,
                    epochs=model_config['epochs'],
                    batch_size=model_config['batch_size'],
                    verbose=1
                )
            
            logger.info("PLSTM-TAL training completed successfully")
            return model, history, best_params
            
        except Exception as e:
            logger.error(f"Error training PLSTM-TAL: {str(e)}")
            raise
    
    def train_benchmark_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Train all benchmark models for comparison
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with benchmark model results
        """
        try:
            logger.info("Training benchmark models...")
            
            # Combine training and validation data for benchmark models
            X_train_combined = np.concatenate([X_train, X_val], axis=0)
            y_train_combined = np.concatenate([y_train, y_val], axis=0)
            
            # Initialize benchmark runner
            runner = BenchmarkModelsRunner(
                sequence_length=X_train.shape[1],
                n_features=X_train.shape[2]
            )
            
            # Train all models
            training_histories = runner.train_all_models(X_train_combined, y_train_combined)
            
            # Evaluate all models
            evaluation_results = runner.evaluate_all_models(X_test, y_test)
            
            # Combine results
            benchmark_results = {}
            for model_name in evaluation_results.keys():
                benchmark_results[model_name] = {
                    'metrics': evaluation_results[model_name],
                    'training_history': training_histories.get(model_name, "No history available")
                }
            
            logger.info("Benchmark models training completed")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Error training benchmark models: {str(e)}")
            raise
    
    def evaluate_model(self, model: PLSTMTAL, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model comprehensively
        
        Args:
            model: Trained PLSTM-TAL model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info("Evaluating model performance...")
            
            # Get predictions
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = y_test.flatten()
            
            # Calculate comprehensive metrics
            metrics = self.evaluator.calculate_all_metrics(y_true, y_pred, y_pred_proba.flatten())
            
            # Generate evaluation report
            report = self.evaluator.generate_evaluation_report(y_true, y_pred, y_pred_proba.flatten(), "PLSTM-TAL")
            
            logger.info(f"Model evaluation completed - Accuracy: {metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model: PLSTMTAL, index_name: str, 
                  best_params: Dict = None, metrics: Dict = None) -> str:
        """
        Save trained model with metadata
        
        Args:
            model: Trained model to save
            index_name: Stock index name
            best_params: Best hyperparameters (optional)
            metrics: Model performance metrics (optional)
            
        Returns:
            Path to saved model
        """
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"plstm_tal_{index_name.lower()}_{timestamp}.h5"
            model_path = os.path.join(MODELS_DIR, model_filename)
            
            # Save model
            model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'index_name': index_name,
                'timestamp': timestamp,
                'model_path': model_path,
                'best_params': best_params,
                'metrics': metrics
            }
            
            metadata_filename = f"plstm_tal_{index_name.lower()}_{timestamp}_metadata.json"
            metadata_path = os.path.join(MODELS_DIR, metadata_filename)
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model and metadata saved: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def train_complete_pipeline(self, index_name: str,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               use_bayesian_opt: bool = True,
                               save_model: bool = True,
                               **config_override) -> Tuple[PLSTMTAL, Dict, Dict, Dict]:
        """
        Run complete training pipeline for a single stock index
        
        Args:
            index_name: Stock index name
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            use_bayesian_opt: Whether to use Bayesian optimization
            save_model: Whether to save the trained model
            **config_override: Configuration parameters to override
            
        Returns:
            Tuple of (model, history, best_params, benchmark_results)
        """
        try:
            logger.info(f"Starting complete training pipeline for {index_name}")
            
            # Step 1: Train PLSTM-TAL model
            logger.info("Step 1: Training PLSTM-TAL model")
            model, history, best_params = self.train_plstm_tal(
                X_train, y_train, X_val, y_val,
                use_bayesian_opt=use_bayesian_opt,
                **config_override
            )
            
            # Step 2: Evaluate PLSTM-TAL model
            logger.info("Step 2: Evaluating PLSTM-TAL model")
            plstm_metrics = self.evaluate_model(model, X_test, y_test)
            
            # Step 3: Train benchmark models
            logger.info("Step 3: Training benchmark models")
            benchmark_results = self.train_benchmark_models(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Add PLSTM-TAL results to benchmark comparison
            benchmark_results['PLSTM-TAL'] = {
                'metrics': plstm_metrics,
                'training_history': history
            }
            
            # Step 4: Compare all models
            logger.info("Step 4: Comparing all models")
            model_metrics = {name: result['metrics'] for name, result in benchmark_results.items() 
                           if isinstance(result['metrics'], dict) and 'error' not in result['metrics']}
            
            comparison_df = self.evaluator.compare_models(model_metrics)
            logger.info("Model comparison completed")
            
            # Step 5: Save model and results
            if save_model:
                logger.info("Step 5: Saving model and results")
                model_path = self.save_model(model, index_name, best_params, plstm_metrics)
            
            # Store results
            self.trained_models[index_name] = model
            self.training_history[index_name] = {
                'plstm_history': history,
                'best_params': best_params,
                'benchmark_results': benchmark_results,
                'comparison': comparison_df
            }
            
            # Log final results
            logger.info(f"Training pipeline completed for {index_name}")
            logger.info(f"PLSTM-TAL Accuracy: {plstm_metrics['accuracy']:.4f}")
            
            if not comparison_df.empty:
                best_model = comparison_df.iloc[0]['Model']
                logger.info(f"Best performing model: {best_model}")
            
            return model, history, best_params, benchmark_results
            
        except Exception as e:
            logger.error(f"Error in complete training pipeline for {index_name}: {str(e)}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of all training results
        
        Returns:
            Training summary dictionary
        """
        try:
            summary = {
                'total_models_trained': len(self.trained_models),
                'indices_processed': list(self.trained_models.keys()),
                'timestamp': datetime.now().isoformat(),
            }
            
            # Add performance summary
            performance_summary = {}
            for index_name, history in self.training_history.items():
                benchmark_results = history.get('benchmark_results', {})
                plstm_metrics = benchmark_results.get('PLSTM-TAL', {}).get('metrics', {})
                
                if plstm_metrics:
                    performance_summary[index_name] = {
                        'accuracy': plstm_metrics.get('accuracy', 0),
                        'f1_score': plstm_metrics.get('f1_score', 0),
                        'auc_roc': plstm_metrics.get('auc_roc', 0),
                        'best_params': history.get('best_params', {})
                    }
            
            summary['performance_summary'] = performance_summary
            
            # Calculate overall statistics
            if performance_summary:
                accuracies = [metrics['accuracy'] for metrics in performance_summary.values()]
                summary['overall_stats'] = {
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating training summary: {str(e)}")
            return {}
    
    def generate_training_report(self) -> str:
        """
        Generate comprehensive training report
        
        Returns:
            Formatted training report
        """
        try:
            summary = self.get_training_summary()
            
            report = f"""
=== PLSTM-TAL Training Report ===
Generated: {summary.get('timestamp', 'Unknown')}

Overview:
- Total models trained: {summary.get('total_models_trained', 0)}
- Stock indices processed: {', '.join(summary.get('indices_processed', []))}

Performance Summary:
"""
            
            performance_summary = summary.get('performance_summary', {})
            for index_name, metrics in performance_summary.items():
                report += f"""
{index_name}:
  - Accuracy: {metrics['accuracy']:.4f}
  - F1-Score: {metrics['f1_score']:.4f}
  - AUC-ROC: {metrics['auc_roc']:.4f}
"""
            
            overall_stats = summary.get('overall_stats', {})
            if overall_stats:
                report += f"""
Overall Statistics:
- Average Accuracy: {overall_stats['avg_accuracy']:.4f} ± {overall_stats['std_accuracy']:.4f}
- Best Performance: {overall_stats['max_accuracy']:.4f}
- Worst Performance: {overall_stats['min_accuracy']:.4f}
"""
            
            report += "\n=== End Report ===\n"
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            return "Error generating report"

def train_all_indices(processed_data: Dict[str, Dict], 
                     use_bayesian_opt: bool = True,
                     save_models: bool = True) -> Tuple[PLSTMTALTrainer, Dict[str, Any]]:
    """
    Convenience function to train PLSTM-TAL for all stock indices
    
    Args:
        processed_data: Dictionary with processed data for all indices
        use_bayesian_opt: Whether to use Bayesian optimization
        save_models: Whether to save trained models
        
    Returns:
        Tuple of (trainer_instance, training_summary)
    """
    trainer = PLSTMTALTrainer()
    
    for index_name, data in processed_data.items():
        try:
            logger.info(f"Training {index_name}...")
            
            model, history, best_params, benchmark_results = trainer.train_complete_pipeline(
                index_name=index_name,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                X_test=data['X_test'],
                y_test=data['y_test'],
                use_bayesian_opt=use_bayesian_opt,
                save_model=save_models
            )
            
            logger.info(f"Successfully trained {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to train {index_name}: {str(e)}")
            continue
    
    # Generate summary
    summary = trainer.get_training_summary()
    
    return trainer, summary

if __name__ == "__main__":
    # Test training pipeline
    from data.preprocessing import preprocess_stock_data
    
    logger.info("Testing PLSTM-TAL training pipeline...")
    
    try:
        # Load and preprocess data
        processed_data, preprocessing_summary = preprocess_stock_data()
        
        # Train models for all indices
        trainer, training_summary = train_all_indices(
            processed_data, 
            use_bayesian_opt=False,  # Disable for testing
            save_models=False
        )
        
        # Generate and print report
        report = trainer.generate_training_report()
        print(report)
        
    except Exception as e:
        logger.error(f"Error in training pipeline test: {str(e)}")
        raise
