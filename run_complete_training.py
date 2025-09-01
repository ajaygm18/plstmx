#!/usr/bin/env python3
"""
Complete Model Training Script - PMC10963254 Implementation
Runs the Full Enhanced Bayesian Optimization and Extended Training phases with MAXIMUM ACCURACY
optimization using unlimited resources and all timeouts removed to achieve 70%+ accuracy target 
and save all results in project folder
"""

import os
import sys
import logging
import json
import pickle
import numpy as np
from datetime import datetime

# Configure GPU settings BEFORE any TensorFlow imports
from utils.gpu_config import ensure_gpu_configured
ensure_gpu_configured()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_training():
    """Run complete training and save all models and results"""
    
    logger.info("🚀 Starting Complete Model Training for PMC10963254 - US INDEX ONLY")
    logger.info("🎯 Target: 70% accuracy with unlimited resources and exhaustive optimization")
    logger.info("⚡ Configuration: 1000 epochs, 500 Bayesian calls, 200 EEMD trials")
    
    try:
        # Import modules
        from data.preprocessing import preprocess_stock_data
        from training.trainer import PLSTMTALTrainer
        from config.settings import EXTENDED_TRAINING_CONFIG, BAYESIAN_OPT_CONFIG, EEMD_CONFIG
        
        # Step 1: Load preprocessed data
        logger.info("📊 Loading preprocessed data...")
        processed_data, preprocessing_summary = preprocess_stock_data()
        
        # Create results directory
        results_dir = "results/complete_training"
        models_dir = "saved_models/plstm_tal"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Step 2: Initialize trainer
        logger.info("🤖 Initializing PLSTM-TAL Trainer...")
        trainer = PLSTMTALTrainer()
        
        # Step 3: Train models for each index
        training_results = {}
        
        for index_name, data in processed_data.items():
            if 'error' in data:
                logger.warning(f"⚠️ Skipping {index_name} due to preprocessing errors")
                continue
                
            logger.info(f"🔄 Training PLSTM-TAL model for {index_name}...")
            
            # Maximum accuracy training configuration - Optimized for 70%+ accuracy
            training_config = {
                'epochs': EXTENDED_TRAINING_CONFIG.get('epochs', 1000),  # Maximum training epochs
                'batch_size': 32,
                'target_accuracy': EXTENDED_TRAINING_CONFIG['target_accuracy'],  # 70% target
                'patience': EXTENDED_TRAINING_CONFIG.get('accuracy_patience', 200),  # Extended patience for higher accuracy
                'enable_checkpoints': True,
                'enable_advanced_optimization': True,  # Enable all advanced optimization techniques
                'use_cosine_annealing': EXTENDED_TRAINING_CONFIG.get('cosine_annealing', True),
                'use_warm_restarts': EXTENDED_TRAINING_CONFIG.get('warm_restarts', True),
                'use_cyclic_lr': EXTENDED_TRAINING_CONFIG.get('cyclic_learning_rate', True),
                'enable_ensemble': EXTENDED_TRAINING_CONFIG.get('ensemble_training', True),
                'enable_data_augmentation': EXTENDED_TRAINING_CONFIG.get('data_augmentation', True)
            }
            
            try:
                # Train model with maximum Bayesian optimization for highest accuracy
                model, history, best_params, benchmark_results = trainer.train_complete_pipeline(
                    index_name=index_name,
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_val=data['X_val'],
                    y_val=data['y_val'],
                    X_test=data['X_test'],
                    y_test=data['y_test'],
                    use_bayesian_opt=True,
                    n_calls=BAYESIAN_OPT_CONFIG.get('n_calls', 500),  # Maximum Bayesian optimization
                    **training_config
                )
                
                # Evaluate final model
                final_metrics = trainer.evaluate_model(model, data['X_test'], data['y_test'])
                
                # Check if target achieved
                target_achieved = final_metrics.get('accuracy', 0) >= training_config['target_accuracy']
                
                # Save model
                model_path = os.path.join(models_dir, f"{index_name}_plstm_tal_model.keras")
                model.save(model_path)
                logger.info(f"💾 Model saved: {model_path}")
                
                # Save training history
                history_path = os.path.join(results_dir, f"{index_name}_training_history.json")
                with open(history_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_history = {}
                    for key, value in history.items():
                        if isinstance(value, np.ndarray):
                            serializable_history[key] = value.tolist()
                        elif isinstance(value, list):
                            serializable_history[key] = value
                        else:
                            serializable_history[key] = str(value)
                    json.dump(serializable_history, f, indent=2)
                
                # Save best parameters
                params_path = os.path.join(results_dir, f"{index_name}_best_params.json")
                with open(params_path, 'w') as f:
                    json.dump(best_params, f, indent=2, default=str)
                
                # Save evaluation metrics
                metrics_path = os.path.join(results_dir, f"{index_name}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(final_metrics, f, indent=2, default=str)
                
                # Store results
                training_results[index_name] = {
                    'model_path': model_path,
                    'final_metrics': final_metrics,
                    'target_achieved': target_achieved,
                    'best_params': best_params,
                    'benchmark_results': benchmark_results,
                    'training_config': training_config
                }
                
                # Log results
                accuracy = final_metrics.get('accuracy', 0)
                if target_achieved:
                    logger.info(f"🎯 {index_name}: SUCCESS! Achieved {accuracy:.4f} accuracy (≥{training_config['target_accuracy']:.1%})")
                else:
                    logger.info(f"⚠️ {index_name}: {accuracy:.4f} accuracy (<{training_config['target_accuracy']:.1%} target)")
                
            except Exception as e:
                logger.error(f"❌ Training failed for {index_name}: {str(e)}")
                training_results[index_name] = {'error': str(e)}
        
        # Step 4: Generate comprehensive results summary with detailed accuracy results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_mode': 'maximum_accuracy_optimization_unlimited_resources',
            'optimization_level': 'maximum',
            'resource_constraints': 'none',
            'total_indices_processed': len([r for r in training_results.values() if 'error' not in r]),
            'successful_trainings': len([r for r in training_results.values() if r.get('target_achieved', False)]),
            'accuracy_results': {},  # Detailed accuracy tracking
            'training_results': training_results,
            'configuration': {
                'extended_training': EXTENDED_TRAINING_CONFIG,
                'bayesian_optimization': BAYESIAN_OPT_CONFIG,
                'eemd_config': EEMD_CONFIG,
                'timeouts_disabled': True,
                'all_features_enabled': True,
                'maximum_accuracy_mode': True,
                'target_accuracy': EXTENDED_TRAINING_CONFIG['target_accuracy']
            }
        }
        
        # Extract detailed accuracy results for each index
        for index_name, result in training_results.items():
            if 'error' not in result and 'final_metrics' in result:
                summary['accuracy_results'][index_name] = {
                    'accuracy': result['final_metrics'].get('accuracy', 0.0),
                    'precision': result['final_metrics'].get('precision', 0.0),
                    'recall': result['final_metrics'].get('recall', 0.0),
                    'f1_score': result['final_metrics'].get('f1_score', 0.0),
                    'auc_roc': result['final_metrics'].get('auc_roc', 0.0),
                    'target_achieved': result.get('target_achieved', False),
                    'target_accuracy': result['training_config']['target_accuracy']
                }
        
        # Calculate overall statistics
        if summary['accuracy_results']:
            accuracies = [r['accuracy'] for r in summary['accuracy_results'].values()]
            summary['overall_accuracy_stats'] = {
                'mean_accuracy': np.mean(accuracies),
                'max_accuracy': np.max(accuracies),
                'min_accuracy': np.min(accuracies),
                'std_accuracy': np.std(accuracies),
                'models_above_target': len([r for r in summary['accuracy_results'].values() if r['target_achieved']]),
                'success_rate': len([r for r in summary['accuracy_results'].values() if r['target_achieved']]) / len(summary['accuracy_results'])
            }
        
        # Save complete summary
        summary_path = os.path.join(results_dir, "complete_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save dedicated accuracy results file for easy access
        accuracy_results_path = os.path.join(results_dir, "accuracy_results.json")
        accuracy_only_summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_mode': 'maximum_accuracy_optimization_unlimited_resources',
            'optimization_level': 'maximum',
            'accuracy_results': summary['accuracy_results'],
            'overall_stats': summary.get('overall_accuracy_stats', {}),
            'configuration_summary': {
                'target_accuracy': EXTENDED_TRAINING_CONFIG['target_accuracy'],  # 70% target
                'epochs_used': EXTENDED_TRAINING_CONFIG['epochs'],  # 1000 epochs
                'bayesian_calls': BAYESIAN_OPT_CONFIG['n_calls'],  # 500 calls
                'eemd_trials': EEMD_CONFIG['trials'],  # 500 trials
                'timeouts_disabled': True,
                'all_features_enabled': True,
                'maximum_accuracy_mode': True,
                'unlimited_resources': True
            }
        }
        with open(accuracy_results_path, 'w') as f:
            json.dump(accuracy_only_summary, f, indent=2, default=str)
        
        logger.info(f"📋 Complete summary saved: {summary_path}")
        logger.info(f"🎯 Accuracy results saved: {accuracy_results_path}")
        
        # Final report with accuracy details
        successful_models = len([r for r in training_results.values() if r.get('target_achieved', False)])
        total_models = len([r for r in training_results.values() if 'error' not in r])
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 MAXIMUM ACCURACY TRAINING FINISHED - UNLIMITED RESOURCES!")
        logger.info(f"🎯 Target Accuracy: {EXTENDED_TRAINING_CONFIG['target_accuracy']:.1%} (70%+)")
        logger.info(f"⚡ Training Configuration: {EXTENDED_TRAINING_CONFIG['epochs']} epochs, {BAYESIAN_OPT_CONFIG['n_calls']} Bayesian calls")
        logger.info(f"✅ Models Successfully Trained: {total_models}")
        logger.info(f"🏆 Target Accuracy Achieved: {successful_models}/{total_models}")
        
        # Log individual accuracy results
        if summary.get('accuracy_results'):
            logger.info("\n📊 ACCURACY RESULTS BY INDEX:")
            for index_name, acc_data in summary['accuracy_results'].items():
                status = "✅" if acc_data['target_achieved'] else "⚠️"
                logger.info(f"  {status} {index_name}: {acc_data['accuracy']:.4f} (target: {acc_data['target_accuracy']:.1%})")
        
        # Log overall statistics
        if summary.get('overall_accuracy_stats'):
            stats = summary['overall_accuracy_stats']
            logger.info(f"\n📈 OVERALL ACCURACY STATISTICS:")
            logger.info(f"  📊 Mean Accuracy: {stats['mean_accuracy']:.4f}")
            logger.info(f"  🏆 Max Accuracy: {stats['max_accuracy']:.4f}")
            logger.info(f"  📉 Min Accuracy: {stats['min_accuracy']:.4f}")
            logger.info(f"  🎯 Success Rate: {stats['success_rate']:.1%}")
        
        logger.info(f"\n💾 Models Saved to: {models_dir}")
        logger.info(f"📊 Results Saved to: {results_dir}")
        logger.info(f"🎯 Accuracy Results: {accuracy_results_path}")
        logger.info("=" * 80)
        
        return True, training_results
        
    except Exception as e:
        logger.error(f"❌ Complete training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, {}

if __name__ == "__main__":
    print("🚀 PMC10963254 Complete Model Training - US INDEX ONLY (70% Target)")
    print("This will train and save models for US stock index only with:")
    print("  • TARGET ACCURACY: 70% (as specified in requirements)")
    print("  • EXTENDED TRAINING: 1000 epochs (increased from 300)")
    print("  • EXHAUSTIVE OPTIMIZATION: 500 Bayesian calls (increased from 200)")
    print("  • ENHANCED SIGNAL PROCESSING: 200 EEMD trials")
    print("  • UNLIMITED RESOURCES: All timeouts removed, unlimited execution time")
    print("  • ADVANCED TECHNIQUES: Cosine annealing, warm restarts, ensemble training")
    print("  • US INDEX ONLY: S&P 500 (^GSPC) data only")
    print("  • COMPREHENSIVE RESULTS: All accuracy metrics saved in project folder")
    print("=" * 80)
    print("⚠️  WARNING: This is resource-intensive training optimized for maximum accuracy!")
    print("🕒 Expected execution time: Several hours to days depending on hardware")
    print("💾 Ensure sufficient disk space for models, checkpoints, and logs")
    print("=" * 80)
    
    success, results = run_complete_training()
    
    if success:
        print("\n🎉 SUCCESS: Maximum accuracy training finished with all results saved!")
        print("🏆 Check results/complete_training/accuracy_results.json for detailed metrics")
        print("💾 Models saved in saved_models/plstm_tal/ directory")
    else:
        print("\n❌ Training encountered issues. Check logs for details.")