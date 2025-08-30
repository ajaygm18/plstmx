"""
Comprehensive evaluation metrics module for PLSTM-TAL model
Implements all metrics mentioned in PMC10963254 research paper
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, log_loss,
    roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for stock market prediction models
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics mentioned in the research paper
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        try:
            # Ensure arrays are flattened and correct type
            y_true = np.array(y_true).flatten().astype(int)
            y_pred = np.array(y_pred).flatten().astype(int)
            
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Confusion matrix for additional metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Negative Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # False Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'specificity': specificity,
                'npv': npv,
                'fpr': fpr,
                'fnr': fnr,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
            
            # Metrics requiring probabilities
            if y_pred_proba is not None:
                y_pred_proba = np.array(y_pred_proba).flatten()
                
                # Ensure probabilities are valid
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                
                # AUC-ROC
                try:
                    auc_roc = roc_auc_score(y_true, y_pred_proba)
                    metrics['auc_roc'] = auc_roc
                except ValueError as e:
                    logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
                    metrics['auc_roc'] = 0.5
                
                # PR-AUC (Area under Precision-Recall curve)
                try:
                    pr_auc = average_precision_score(y_true, y_pred_proba)
                    metrics['pr_auc'] = pr_auc
                except ValueError as e:
                    logger.warning(f"Could not calculate PR-AUC: {str(e)}")
                    metrics['pr_auc'] = np.mean(y_true)  # Random baseline
                
                # Log Loss
                try:
                    logloss = log_loss(y_true, y_pred_proba)
                    metrics['log_loss'] = logloss
                except ValueError as e:
                    logger.warning(f"Could not calculate log loss: {str(e)}")
                    metrics['log_loss'] = np.inf
            else:
                # Set default values when probabilities are not available
                metrics.update({
                    'auc_roc': 0.5,
                    'pr_auc': np.mean(y_true),
                    'log_loss': np.inf
                })
            
            # Store metrics history
            self.metrics_history.append(metrics.copy())
            
            logger.info(f"Calculated metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Confusion matrix as numpy array
        """
        try:
            y_true = np.array(y_true).flatten().astype(int)
            y_pred = np.array(y_pred).flatten().astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            return cm
            
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {str(e)}")
            raise
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Classification report as string
        """
        try:
            y_true = np.array(y_true).flatten().astype(int)
            y_pred = np.array(y_pred).flatten().astype(int)
            
            target_names = ['Down', 'Up']  # Stock price movement directions
            report = classification_report(y_true, y_pred, target_names=target_names)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise
    
    def get_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with ROC curve data
        """
        try:
            y_true = np.array(y_true).flatten().astype(int)
            y_pred_proba = np.array(y_pred_proba).flatten()
            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROC curve: {str(e)}")
            raise
    
    def get_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get Precision-Recall curve data
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with PR curve data
        """
        try:
            y_true = np.array(y_true).flatten().astype(int)
            y_pred_proba = np.array(y_pred_proba).flatten()
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            auc = average_precision_score(y_true, y_pred_proba)
            
            return {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
                'auc': auc
            }
            
        except Exception as e:
            logger.error(f"Error calculating Precision-Recall curve: {str(e)}")
            raise
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                returns: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate trading-specific metrics
        
        Args:
            y_true: True binary labels (1 for up, 0 for down)
            y_pred: Predicted binary labels
            returns: Actual returns for the period (optional)
            
        Returns:
            Dictionary with trading metrics
        """
        try:
            y_true = np.array(y_true).flatten().astype(int)
            y_pred = np.array(y_pred).flatten().astype(int)
            
            # Basic trading metrics
            total_trades = len(y_pred)
            correct_predictions = np.sum(y_true == y_pred)
            
            # Directional accuracy
            directional_accuracy = correct_predictions / total_trades
            
            # Up predictions accuracy
            up_predictions = np.sum(y_pred == 1)
            up_correct = np.sum((y_true == 1) & (y_pred == 1))
            up_accuracy = up_correct / up_predictions if up_predictions > 0 else 0
            
            # Down predictions accuracy
            down_predictions = np.sum(y_pred == 0)
            down_correct = np.sum((y_true == 0) & (y_pred == 0))
            down_accuracy = down_correct / down_predictions if down_predictions > 0 else 0
            
            trading_metrics = {
                'directional_accuracy': directional_accuracy,
                'total_trades': total_trades,
                'correct_predictions': correct_predictions,
                'up_predictions': up_predictions,
                'up_accuracy': up_accuracy,
                'down_predictions': down_predictions,
                'down_accuracy': down_accuracy,
                'up_prediction_rate': up_predictions / total_trades,
                'down_prediction_rate': down_predictions / total_trades
            }
            
            # If returns are provided, calculate return-based metrics
            if returns is not None:
                returns = np.array(returns).flatten()
                
                # Strategy returns (assuming we trade based on predictions)
                strategy_returns = np.where(y_pred == 1, returns, 0)  # Only trade on up predictions
                
                # Cumulative returns
                cumulative_returns = np.cumprod(1 + strategy_returns) - 1
                
                # Additional metrics
                total_return = cumulative_returns[-1]
                annualized_return = (1 + total_return) ** (252 / len(returns)) - 1  # Assuming daily data
                volatility = np.std(strategy_returns) * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                trading_metrics.update({
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': self._calculate_max_drawdown(cumulative_returns)
                })
            
            return trading_metrics
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {str(e)}")
            raise
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            cumulative_returns: Cumulative returns array
            
        Returns:
            Maximum drawdown value
        """
        try:
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            
        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []
            
            for model_name, metrics in model_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': metrics.get('accuracy', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1-Score': metrics.get('f1_score', 0),
                        'AUC-ROC': metrics.get('auc_roc', 0),
                        'PR-AUC': metrics.get('pr_auc', 0),
                        'MCC': metrics.get('mcc', 0)
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                
                # Add rankings
                metrics_to_rank = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'MCC']
                for metric in metrics_to_rank:
                    if metric in df.columns:
                        df[f'{metric}_Rank'] = df[metric].rank(ascending=False, method='min')
                
                # Calculate average rank
                rank_columns = [col for col in df.columns if col.endswith('_Rank')]
                df['Average_Rank'] = df[rank_columns].mean(axis=1)
                
                # Sort by average rank
                df = df.sort_values('Average_Rank').reset_index(drop=True)
                
                return df
            else:
                logger.warning("No valid model results for comparison")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray = None, 
                                 model_name: str = "Model") -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        try:
            # Calculate all metrics
            metrics = self.calculate_all_metrics(y_true, y_pred, y_pred_proba)
            
            # Generate report
            report = f"""
=== {model_name} Evaluation Report ===

Basic Classification Metrics:
- Accuracy:     {metrics['accuracy']:.4f}
- Precision:    {metrics['precision']:.4f}
- Recall:       {metrics['recall']:.4f}
- F1-Score:     {metrics['f1_score']:.4f}
- Specificity:  {metrics['specificity']:.4f}

Advanced Metrics:
- AUC-ROC:      {metrics['auc_roc']:.4f}
- PR-AUC:       {metrics['pr_auc']:.4f}
- MCC:          {metrics['mcc']:.4f}
- Log Loss:     {metrics['log_loss']:.4f}

Confusion Matrix:
- True Positives:  {metrics['true_positives']}
- True Negatives:  {metrics['true_negatives']}
- False Positives: {metrics['false_positives']}
- False Negatives: {metrics['false_negatives']}

Error Rates:
- False Positive Rate: {metrics['fpr']:.4f}
- False Negative Rate: {metrics['fnr']:.4f}

Classification Report:
{self.get_classification_report(y_true, y_pred)}
            """
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all calculated metrics
        
        Returns:
            Summary statistics of metrics
        """
        if not self.metrics_history:
            return {}
        
        try:
            metrics_df = pd.DataFrame(self.metrics_history)
            
            summary = {
                'total_evaluations': len(self.metrics_history),
                'mean_metrics': metrics_df.mean().to_dict(),
                'std_metrics': metrics_df.std().to_dict(),
                'best_accuracy': metrics_df['accuracy'].max(),
                'best_f1': metrics_df['f1_score'].max(),
                'best_auc_roc': metrics_df['auc_roc'].max()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating metrics summary: {str(e)}")
            return {}

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: np.ndarray = None,
                             model_name: str = "Model") -> Tuple[Dict[str, float], str]:
    """
    Convenience function to evaluate model performance
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities (optional)
        model_name: Name of the model
        
    Returns:
        Tuple of (metrics_dict, evaluation_report)
    """
    evaluator = EvaluationMetrics()
    
    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    # Generate report
    report = evaluator.generate_evaluation_report(y_true, y_pred, y_pred_proba, model_name)
    
    return metrics, report

if __name__ == "__main__":
    # Test evaluation metrics
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Test metrics calculation
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    print("Calculated metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test report generation
    report = evaluator.generate_evaluation_report(y_true, y_pred, y_pred_proba, "Test Model")
    print("\nEvaluation Report:")
    print(report)
