"""
Main Streamlit application for PLSTM-TAL Stock Market Prediction
Implementation of research paper PMC10963254
"""

# Configure GPU settings BEFORE any TensorFlow imports
from utils.gpu_config import ensure_gpu_configured
ensure_gpu_configured()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import os
import time
from datetime import datetime

# Import custom modules
from config.settings import STOCK_INDICES, PLSTM_CONFIG
from data.preprocessing import preprocess_stock_data
from training.trainer import PLSTMTALTrainer
from utils.evaluation_metrics import EvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PLSTM-TAL Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    
    st.title("📈 PLSTM-TAL Stock Market Prediction")
    st.markdown("*Implementation of PMC10963254 Research Paper*")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Data Processing", "Model Training", "Evaluation", "Results"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Data Processing":
        show_data_processing()
    elif page == "Model Training":
        show_model_training()
    elif page == "Evaluation":
        show_evaluation()
    elif page == "Results":
        show_results()

def show_overview():
    """Show overview of the PLSTM-TAL model and research paper"""
    
    st.header("Research Paper Overview")
    st.write("**Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.write("**PLSTM-TAL** combines:")
        st.write("- 🔍 **Peephole LSTM**: Enhanced LSTM with peephole connections")
        st.write("- ⏰ **Temporal Attention Layer**: Time-aware attention mechanism")
        st.write("- 🎯 **Binary Classification**: Predicts price movement direction")
        
        st.subheader("Data Sources")
        st.write("**Stock Indices (2005-2022):**")
        for name, symbol in STOCK_INDICES.items():
            st.write(f"- **{name}**: {symbol}")
    
    with col2:
        st.subheader("Processing Pipeline")
        st.write("1. **Data Collection**: Historical OHLCV data")
        st.write("2. **Technical Indicators**: 40 indicators calculated")
        st.write("3. **EEMD Filtering**: Noise reduction via decomposition")
        st.write("4. **Feature Extraction**: Contractive Autoencoder")
        st.write("5. **Hyperparameter Tuning**: Bayesian optimization")
        st.write("6. **Model Training**: PLSTM-TAL architecture")
        
        st.subheader("Target Accuracy")
        target_accuracy = {
            "U.K.": "96%",
            "China": "88%",
            "U.S.": "85%",
            "India": "85%"
        }
        for country, acc in target_accuracy.items():
            st.write(f"- **{country}**: {acc}")

def show_data_processing():
    """Show data processing pipeline and results"""
    
    st.header("📊 Data Processing Pipeline")
    
    # Processing controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Load Stock Data", use_container_width=True):
            with st.spinner("Loading stock market data..."):
                try:
                    processed_data, summary = preprocess_stock_data()
                    st.session_state.processed_data = processed_data
                    st.session_state.preprocessing_summary = summary
                    st.success("Data processing completed!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    logger.error(f"Data processing error: {str(e)}")
    
    with col2:
        show_details = st.checkbox("Show Processing Details")
    
    with col3:
        if st.button("📋 View Summary", use_container_width=True):
            if 'preprocessing_summary' in st.session_state:
                st.session_state.show_summary = True
    
    # Display processing results
    if 'preprocessing_summary' in st.session_state:
        
        if st.session_state.get('show_summary', False):
            st.subheader("📈 Data Summary")
            
            summary_df = pd.DataFrame(st.session_state.preprocessing_summary).T
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Training Samples', 'Feature Dimensions', 'Class Distribution', 'Noise Reduction'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            indices = list(summary_df.index)
            
            # Training samples
            fig.add_trace(
                go.Bar(x=indices, y=summary_df['train_samples'], name='Train Samples'),
                row=1, col=1
            )
            
            # Feature dimensions
            fig.add_trace(
                go.Bar(x=indices, y=summary_df['feature_dim'], name='Feature Dim', marker_color='orange'),
                row=1, col=2
            )
            
            # Class distribution
            fig.add_trace(
                go.Bar(x=indices, y=summary_df['train_up_ratio']*100, name='Up %', marker_color='green'),
                row=2, col=1
            )
            
            # Noise reduction
            fig.add_trace(
                go.Bar(x=indices, y=summary_df['noise_reduction'], name='Noise Reduction %', marker_color='red'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Data Processing Results")
            st.plotly_chart(fig, use_container_width=True)
        
        if show_details:
            st.subheader("🔍 Processing Details")
            
            for index_name in STOCK_INDICES.keys():
                if index_name in st.session_state.preprocessing_summary:
                    with st.expander(f"📊 {index_name} Details"):
                        stats = st.session_state.preprocessing_summary[index_name]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Training Samples", f"{stats['train_samples']:,}")
                            st.metric("Validation Samples", f"{stats['val_samples']:,}")
                        
                        with col2:
                            st.metric("Test Samples", f"{stats['test_samples']:,}")
                            st.metric("Feature Dimension", stats['feature_dim'])
                        
                        with col3:
                            st.metric("Sequence Length", stats['sequence_length'])
                            st.metric("Noise Reduction", f"{stats['noise_reduction']:.2f}%")

def show_model_training():
    """Show model training interface and progress"""
    
    st.header("🤖 Model Training")
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please process data first in the Data Processing section.")
        return
    
    # Enhanced Training Configuration for PMC10963254
    st.subheader("Extended Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_indices = st.multiselect(
            "Select Indices to Train",
            options=list(st.session_state.processed_data.keys()),
            default=list(st.session_state.processed_data.keys())
        )
        
        use_bayesian_opt = st.checkbox("Use Enhanced Bayesian Optimization", value=True,
            help="Uses 200+ optimization calls for thorough hyperparameter search")
        
        extended_training = st.checkbox("Enable Extended Training", value=True,
            help="Enables extended training with target accuracy of 70%+")
        
    with col2:
        # Enhanced training parameters
        if extended_training:
            epochs = st.slider("Training Epochs (Extended)", 200, 500, 300,
                help="Extended training epochs for better accuracy")
            bayesian_calls = st.slider("Bayesian Optimization Calls", 100, 300, 200,
                help="More calls = better hyperparameter optimization")
        else:
            epochs = st.slider("Training Epochs", 10, 200, PLSTM_CONFIG['epochs'])
            bayesian_calls = 50
            
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    # Extended training options
    if extended_training:
        st.subheader("🎯 Extended Training Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_accuracy = st.slider("Target Accuracy", 0.6, 0.95, 0.70, 0.01,
                help="Training will aim to achieve this accuracy (70%+ as per PMC10963254)")
            
        with col2:
            early_stopping_patience = st.slider("Early Stopping Patience", 30, 100, 50,
                help="Epochs to wait for improvement before stopping")
                
        with col3:
            enable_checkpoints = st.checkbox("Enable Checkpoints", value=True,
                help="Save model checkpoints during training")
        
        # Show estimated training time
        estimated_time = epochs * bayesian_calls * 0.5 / 60  # Rough estimate in minutes
        st.info(f"⏱️ Estimated training time: {estimated_time:.1f} minutes "
                f"(Timeouts disabled for extended training)")
        
        st.warning(f"🎯 **Target**: Achieve {target_accuracy:.1%} accuracy as per PMC10963254 research paper")
    
    # Training controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Enhanced Training", use_container_width=True):
            if selected_indices:
                training_config = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'use_bayesian_opt': use_bayesian_opt,
                    'bayesian_calls': bayesian_calls if use_bayesian_opt else None,
                    'extended_training': extended_training,
                    'target_accuracy': target_accuracy if extended_training else 0.7,
                    'patience': early_stopping_patience if extended_training else 15,
                    'enable_checkpoints': enable_checkpoints if extended_training else False
                }
                start_training(selected_indices, training_config)
            else:
                st.error("Please select at least one index to train.")
    
    with col2:
        if st.button("⏹️ Stop Training", use_container_width=True):
            st.session_state.stop_training = True
            st.warning("Training stop requested...")
    
    with col3:
        if st.button("📊 View Progress", use_container_width=True):
            show_training_progress()
    
    with col2:
        if st.button("⏹️ Stop Training", use_container_width=True):
            st.session_state.stop_training = True
            st.warning("Training stop requested...")
    
    with col3:
        if st.button("📊 View Progress", use_container_width=True):
            show_training_progress()
    
    # Enhanced display of training status
    if 'training_status' in st.session_state:
        st.subheader("📊 Enhanced Training Status")
        
        for index_name, status in st.session_state.training_status.items():
            with st.expander(f"📈 {index_name} Training Results"):
                if status.get('completed', False):
                    # Enhanced completion status
                    target_achieved = status.get('target_achieved', False)
                    extended_training = status.get('extended_training', False)
                    training_time = status.get('training_time', 0)
                    
                    if target_achieved:
                        st.success(f"🎯 SUCCESS: Target accuracy achieved!")
                    else:
                        st.warning(f"⚠️ Target accuracy not achieved")
                    
                    if extended_training:
                        st.info(f"⚡ Extended training completed in {training_time:.1f}s")
                    else:
                        st.info(f"⏱️ Standard training completed in {training_time:.1f}s")
                    
                    metrics = status.get('final_metrics', {})
                    target_acc = status.get('target_accuracy', 0.7)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        accuracy = metrics.get('accuracy', 0)
                        delta_color = "normal" if accuracy >= target_acc else "inverse"
                        st.metric("Accuracy", f"{accuracy:.4f}", 
                                delta=f"Target: {target_acc:.1%}", delta_color=delta_color)
                    with col2:
                        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
                    with col5:
                        auc_score = metrics.get('auc_roc', 0)
                        st.metric("AUC-ROC", f"{auc_score:.4f}")
                
                elif status.get('training', False):
                    st.info(f"🔄 Enhanced training in progress...")
                    
                    # Enhanced progress information
                    current_epoch = status.get('current_epoch', 0)
                    total_epochs = status.get('total_epochs', 1)
                    target_acc = status.get('target_accuracy', 0.7)
                    extended = status.get('extended_training', False)
                    bayesian = status.get('bayesian_optimization', False)
                    
                    progress = current_epoch / total_epochs
                    st.progress(progress)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Epoch {current_epoch}/{total_epochs}")
                        st.write(f"🎯 Target: {target_acc:.1%}")
                    with col2:
                        if extended:
                            st.write("⚡ Extended Training")
                        if bayesian:
                            st.write("🔧 Bayesian Optimization")
                
                else:
                    st.info("⏳ Waiting to start enhanced training...")

def start_training(selected_indices, training_config):
    """Start enhanced training process for selected indices with PMC10963254 configuration"""
    
    st.session_state.training_status = {}
    st.session_state.trained_models = {}
    st.session_state.stop_training = False
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        from config.settings import EXTENDED_TRAINING_CONFIG
        
        # Enhanced trainer initialization
        trainer = PLSTMTALTrainer()
        
        # Extract training configuration
        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        use_bayesian_opt = training_config['use_bayesian_opt']
        bayesian_calls = training_config.get('bayesian_calls', 200)
        extended_training = training_config.get('extended_training', True)
        target_accuracy = training_config.get('target_accuracy', 0.7)
        patience = training_config.get('patience', 50)
        enable_checkpoints = training_config.get('enable_checkpoints', True)
        
        status_placeholder.info(f"🎯 Starting enhanced training with target accuracy: {target_accuracy:.1%}")
        
        if extended_training:
            status_placeholder.info(f"⚡ Extended training enabled: {epochs} epochs, "
                                  f"{bayesian_calls if use_bayesian_opt else 'no'} Bayesian calls")
        
        for i, index_name in enumerate(selected_indices):
            if st.session_state.get('stop_training', False):
                break
            
            progress_placeholder.progress((i) / len(selected_indices))
            status_placeholder.info(f"🔄 Training {index_name} with enhanced configuration...")
            
            # Update status with enhanced information
            st.session_state.training_status[index_name] = {
                'training': True,
                'current_epoch': 0,
                'total_epochs': epochs,
                'target_accuracy': target_accuracy,
                'extended_training': extended_training,
                'bayesian_optimization': use_bayesian_opt,
                'start_time': time.time()
            }
            
            # Get processed data for this index
            data = st.session_state.processed_data[index_name]
            
            # Enhanced training configuration
            enhanced_config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'target_accuracy': target_accuracy,
                'patience': patience,
                'enable_checkpoints': enable_checkpoints
            }
            
            if use_bayesian_opt:
                enhanced_config['n_calls'] = bayesian_calls
            
            # Train model with enhanced configuration
            model, history, best_params, benchmark_results = trainer.train_complete_pipeline(
                index_name=index_name,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                X_test=data['X_test'],
                y_test=data['y_test'],
                use_bayesian_opt=use_bayesian_opt,
                **enhanced_config
            )
            
            # Store enhanced results
            training_time = time.time() - st.session_state.training_status[index_name]['start_time']
            
            st.session_state.trained_models[index_name] = {
                'model': model,
                'history': history,
                'best_params': best_params,
                'benchmark_results': benchmark_results,
                'training_config': enhanced_config,
                'training_time': training_time
            }
            
            # Evaluate model and check if target was achieved
            final_metrics = trainer.evaluate_model(model, data['X_test'], data['y_test'])
            target_achieved = final_metrics.get('accuracy', 0) >= target_accuracy
            
            # Update status with enhanced information
            st.session_state.training_status[index_name] = {
                'completed': True,
                'final_metrics': final_metrics,
                'target_achieved': target_achieved,
                'training_time': training_time,
                'extended_training': extended_training
            }
            
            # Show achievement status
            if target_achieved:
                status_placeholder.success(f"🎯 {index_name}: Target accuracy {target_accuracy:.1%} achieved! "
                                         f"Final accuracy: {final_metrics.get('accuracy', 0):.4f}")
            else:
                status_placeholder.warning(f"⚠️ {index_name}: Target accuracy not achieved. "
                                         f"Final accuracy: {final_metrics.get('accuracy', 0):.4f}")
        
        progress_placeholder.progress(1.0)
        
        # Final summary
        total_indices = len(selected_indices)
        successful_indices = sum(1 for status in st.session_state.training_status.values() 
                               if status.get('target_achieved', False))
        
        if successful_indices == total_indices:
            status_placeholder.success(f"🎉 SUCCESS! All {total_indices} indices achieved target accuracy!")
        elif successful_indices > 0:
            status_placeholder.info(f"✅ {successful_indices}/{total_indices} indices achieved target accuracy")
        else:
            status_placeholder.warning(f"⚠️ No indices achieved target accuracy. Consider extended training.")
        
    except Exception as e:
        status_placeholder.error(f"❌ Enhanced training failed: {str(e)}")
        logger.error(f"Enhanced training error: {str(e)}")

def show_training_progress():
    """Show detailed training progress"""
    
    if 'trained_models' in st.session_state:
        st.subheader("📊 Training Progress Details")
        
        for index_name, results in st.session_state.trained_models.items():
            with st.expander(f"📈 {index_name} Training History"):
                
                history = results.get('history', {})
                
                if history:
                    # Plot training history
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Loss', 'Accuracy', 'Precision', 'Recall']
                    )
                    
                    epochs = range(1, len(history['loss']) + 1)
                    
                    # Loss
                    fig.add_trace(go.Scatter(x=list(epochs), y=history['loss'], name='Train Loss'), row=1, col=1)
                    if 'val_loss' in history:
                        fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], name='Val Loss'), row=1, col=1)
                    
                    # Accuracy
                    fig.add_trace(go.Scatter(x=list(epochs), y=history['accuracy'], name='Train Acc'), row=1, col=2)
                    if 'val_accuracy' in history:
                        fig.add_trace(go.Scatter(x=list(epochs), y=history['val_accuracy'], name='Val Acc'), row=1, col=2)
                    
                    # Precision
                    fig.add_trace(go.Scatter(x=list(epochs), y=history['precision'], name='Train Prec'), row=2, col=1)
                    if 'val_precision' in history:
                        fig.add_trace(go.Scatter(x=list(epochs), y=history['val_precision'], name='Val Prec'), row=2, col=1)
                    
                    # Recall
                    fig.add_trace(go.Scatter(x=list(epochs), y=history['recall'], name='Train Rec'), row=2, col=2)
                    if 'val_recall' in history:
                        fig.add_trace(go.Scatter(x=list(epochs), y=history['val_recall'], name='Val Rec'), row=2, col=2)
                    
                    fig.update_layout(height=600, title_text=f"{index_name} Training History")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show best parameters if available
                if 'best_params' in results and results['best_params']:
                    st.write("**Optimized Hyperparameters:**")
                    params_df = pd.DataFrame([results['best_params']]).T
                    params_df.columns = ['Value']
                    st.dataframe(params_df)

def show_evaluation():
    """Show model evaluation and comparison with benchmarks"""
    
    st.header("📊 Model Evaluation")
    
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    # Evaluation controls
    col1, col2 = st.columns(2)
    
    with col1:
        selected_index = st.selectbox(
            "Select Index for Evaluation",
            options=list(st.session_state.trained_models.keys())
        )
    
    with col2:
        evaluation_type = st.selectbox(
            "Evaluation Type",
            ["Model Comparison", "Detailed Metrics", "Confusion Matrix", "ROC Curves"]
        )
    
    if selected_index and selected_index in st.session_state.trained_models:
        results = st.session_state.trained_models[selected_index]
        
        if evaluation_type == "Model Comparison":
            show_model_comparison(selected_index, results)
        elif evaluation_type == "Detailed Metrics":
            show_detailed_metrics(selected_index, results)
        elif evaluation_type == "Confusion Matrix":
            show_confusion_matrix(selected_index, results)
        elif evaluation_type == "ROC Curves":
            show_roc_curves(selected_index, results)

def show_model_comparison(index_name, results):
    """Show comparison between PLSTM-TAL and benchmark models"""
    
    st.subheader(f"🏆 Model Comparison - {index_name}")
    
    benchmark_results = results.get('benchmark_results', {})
    
    if benchmark_results:
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, metrics in benchmark_results.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=comparison_df['Model'],
                y=comparison_df[metric],
                name=metric
            ))
        
        fig.update_layout(
            title=f"Model Performance Comparison - {index_name}",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight best performing model
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = comparison_df['Accuracy'].max()
        
        st.success(f"🏆 Best performing model: **{best_model}** with {best_accuracy:.4f} accuracy")
    
    else:
        st.warning("No benchmark results available for comparison.")

def show_detailed_metrics(index_name, results):
    """Show detailed evaluation metrics"""
    
    st.subheader(f"📈 Detailed Metrics - {index_name}")
    
    # Get test data
    if 'processed_data' in st.session_state and index_name in st.session_state.processed_data:
        data = st.session_state.processed_data[index_name]
        model = results['model']
        
        # Calculate comprehensive metrics
        evaluator = EvaluationMetrics()
        
        # Get predictions
        y_pred_prob = model.predict(data['X_test'])
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = data['y_test'].flatten()
        
        # Calculate all metrics
        metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_pred_prob.flatten())
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        with col2:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        
        with col3:
            st.metric("MCC", f"{metrics['mcc']:.4f}")
            st.metric("Log Loss", f"{metrics['log_loss']:.4f}")
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
        
        # Show classification report
        st.subheader("📋 Classification Report")
        
        report = evaluator.get_classification_report(y_true, y_pred)
        st.text(report)

def show_confusion_matrix(index_name, results):
    """Show confusion matrix visualization"""
    
    st.subheader(f"🎯 Confusion Matrix - {index_name}")
    
    if 'processed_data' in st.session_state and index_name in st.session_state.processed_data:
        data = st.session_state.processed_data[index_name]
        model = results['model']
        
        # Get predictions
        y_pred_prob = model.predict(data['X_test'])
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = data['y_test'].flatten()
        
        # Calculate confusion matrix
        evaluator = EvaluationMetrics()
        cm = evaluator.get_confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix - {index_name}",
            labels=dict(x="Predicted", y="Actual"),
            x=['Down', 'Up'],
            y=['Down', 'Up']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confusion Matrix Values:**")
            st.write(f"True Negatives (TN): {tn}")
            st.write(f"False Positives (FP): {fp}")
            st.write(f"False Negatives (FN): {fn}")
            st.write(f"True Positives (TP): {tp}")
        
        with col2:
            st.write("**Derived Metrics:**")
            st.write(f"Sensitivity (Recall): {tp/(tp+fn):.4f}")
            st.write(f"Specificity: {tn/(tn+fp):.4f}")
            st.write(f"Positive Predictive Value: {tp/(tp+fp):.4f}")
            st.write(f"Negative Predictive Value: {tn/(tn+fn):.4f}")

def show_roc_curves(index_name, results):
    """Show ROC and PR curves"""
    
    st.subheader(f"📈 ROC and PR Curves - {index_name}")
    
    if 'processed_data' in st.session_state and index_name in st.session_state.processed_data:
        data = st.session_state.processed_data[index_name]
        model = results['model']
        
        # Get predictions
        y_pred_prob = model.predict(data['X_test'])
        y_true = data['y_test'].flatten()
        
        # Calculate curves
        evaluator = EvaluationMetrics()
        roc_data = evaluator.get_roc_curve(y_true, y_pred_prob.flatten())
        pr_data = evaluator.get_precision_recall_curve(y_true, y_pred_prob.flatten())
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve
            fig_roc = go.Figure()
            
            fig_roc.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f'ROC Curve (AUC = {roc_data["auc"]:.4f})',
                line=dict(color='blue', width=2)
            ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            # Precision-Recall Curve
            fig_pr = go.Figure()
            
            fig_pr.add_trace(go.Scatter(
                x=pr_data['recall'],
                y=pr_data['precision'],
                mode='lines',
                name=f'PR Curve (AUC = {pr_data["auc"]:.4f})',
                line=dict(color='green', width=2)
            ))
            
            fig_pr.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                showlegend=True
            )
            
            st.plotly_chart(fig_pr, use_container_width=True)

def show_results():
    """Show final results and comparison with research paper targets"""
    
    st.header("🎯 Final Results")
    
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train models first to see results.")
        return
    
    # Target accuracies from research paper
    target_accuracies = {
        'UK': 0.96,
        'China': 0.88,
        'US': 0.85,
        'India': 0.85
    }
    
    # Collect results
    results_data = []
    
    for index_name in st.session_state.trained_models.keys():
        results = st.session_state.trained_models[index_name]
        
        # Get test accuracy
        if 'processed_data' in st.session_state and index_name in st.session_state.processed_data:
            data = st.session_state.processed_data[index_name]
            model = results['model']
            
            # Evaluate model
            test_metrics = model.evaluate(data['X_test'], data['y_test'], verbose=0)
            achieved_accuracy = test_metrics[1]  # Accuracy is second metric
            
            target_accuracy = target_accuracies.get(index_name, 0.85)
            
            results_data.append({
                'Index': index_name,
                'Target Accuracy': f"{target_accuracy:.1%}",
                'Achieved Accuracy': f"{achieved_accuracy:.1%}",
                'Difference': f"{achieved_accuracy - target_accuracy:+.1%}",
                'Status': '✅ Achieved' if achieved_accuracy >= target_accuracy else '❌ Below Target'
            })
    
    if results_data:
        st.subheader("📊 Accuracy Comparison with Research Paper")
        
        results_df = pd.DataFrame(results_data)
        
        # Style the dataframe
        def style_status(val):
            if '✅' in val:
                return 'background-color: #d4edda; color: #155724'
            elif '❌' in val:
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled_df = results_df.style.applymap(style_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary statistics
        achieved_accuracies = [float(acc.strip('%'))/100 for acc in results_df['Achieved Accuracy']]
        target_accuracies_list = [float(acc.strip('%'))/100 for acc in results_df['Target Accuracy']]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_achieved = np.mean(achieved_accuracies)
            st.metric("Average Achieved Accuracy", f"{avg_achieved:.1%}")
        
        with col2:
            avg_target = np.mean(target_accuracies_list)
            st.metric("Average Target Accuracy", f"{avg_target:.1%}")
        
        with col3:
            overall_diff = avg_achieved - avg_target
            st.metric("Overall Difference", f"{overall_diff:+.1%}")
        
        # Visualization
        st.subheader("📈 Performance Visualization")
        
        fig = go.Figure()
        
        indices = results_df['Index'].tolist()
        
        fig.add_trace(go.Bar(
            x=indices,
            y=[float(acc.strip('%'))/100 for acc in results_df['Target Accuracy']],
            name='Target Accuracy',
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            x=indices,
            y=achieved_accuracies,
            name='Achieved Accuracy',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Target vs Achieved Accuracy by Index',
            xaxis_title='Stock Index',
            yaxis_title='Accuracy',
            barmode='group',
            height=500,
            yaxis_tickformat='.1%'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model architecture summary
        st.subheader("🏗️ Model Architecture Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**PLSTM-TAL Components:**")
            st.write("- Peephole LSTM layers with cell state connections")
            st.write("- Temporal Attention Layer for time-aware feature selection")
            st.write("- Dense layers for classification")
            st.write("- Binary sigmoid output for direction prediction")
        
        with col2:
            st.write("**Data Processing Pipeline:**")
            st.write("- EEMD decomposition for noise reduction")
            st.write("- 40 technical indicators calculation")
            st.write("- Contractive Autoencoder for feature extraction")
            st.write("- Bayesian optimization for hyperparameter tuning")
        
        # Research paper citation
        st.subheader("📚 Research Paper Reference")
        st.info(
            "**Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities**\n\n"
            "Saima Latif, Nadeem Javaid, Faheem Aslam, Abdulaziz Aldegheishem, Nabil Alrajeh, Safdar Hussain Bouk\n\n"
            "Heliyon, Volume 10, Issue 6, 2024, e27747\n\n"
            "DOI: https://doi.org/10.1016/j.heliyon.2024.e27747\n\n"
            "PMC: PMC10963254"
        )
    
    else:
        st.info("No results available yet. Please train the models first.")

if __name__ == "__main__":
    main()
