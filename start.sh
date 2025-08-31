#!/bin/bash
# 
# PLSTM-TAL Stock Market Prediction Project
# Quick Start Guide
#

echo "🚀 PLSTM-TAL Stock Market Prediction Project"
echo "=============================================="
echo ""
echo "This project implements the PLSTM-TAL model from research paper PMC10963254"
echo "for stock market price movement prediction."
echo ""

# Check if running with mock or real dependencies
if command -v streamlit &> /dev/null; then
    echo "✅ Real dependencies detected!"
    echo ""
    echo "🎯 Running with Full Dependencies:"
    echo "   streamlit run app.py"
    echo ""
    echo "📊 Alternative: Run data pipeline test:"
    echo "   python3 test_pipeline.py"
    echo ""
else
    echo "⚠️  Dependencies not installed. Running with mock dependencies..."
    echo ""
    echo "🎯 Run Streamlit Interface (Mock Mode):"
    echo "   python3 run_app.py"
    echo ""
    echo "📊 Test Data Processing Pipeline:"
    echo "   python3 test_pipeline.py"
    echo ""
    echo "🔧 Install Real Dependencies:"
    echo "   pip install -r requirements.txt"
    echo "   streamlit run app.py"
    echo ""
fi

echo "📁 Project Structure:"
echo "   - app.py                 # Main Streamlit application"
echo "   - data/                  # Data loading and preprocessing"
echo "   - models/                # PLSTM-TAL and benchmark models"
echo "   - training/              # Training pipeline"
echo "   - utils/                 # Evaluation metrics and utilities"
echo "   - mocks.py               # Fallback dependencies"
echo ""

echo "🎯 Supported Stock Indices:"
echo "   - US: S&P 500 (^GSPC)"
echo "   - UK: FTSE 100 (^FTSE)"
echo "   - China: SSE Composite (000001.SS)"
echo "   - India: NIFTY 50 (^NSEI)"
echo ""

echo "✨ Model Features:"
echo "   - Peephole LSTM with Temporal Attention"
echo "   - 40 Technical Indicators"
echo "   - EEMD Noise Filtering"
echo "   - Contractive Autoencoder Feature Extraction"
echo "   - Bayesian Hyperparameter Optimization"
echo ""

echo "Choose an option to run:"
echo "1. Run Streamlit Interface (Mock Mode): python3 run_app.py"
echo "2. Test Data Pipeline: python3 test_pipeline.py"
echo "3. Install Dependencies: pip install -r requirements.txt"
echo ""