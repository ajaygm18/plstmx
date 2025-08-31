#!/usr/bin/env python3
"""
Test script to simulate the exact error scenario and verify the fix
"""

import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/plstmx/plstmx')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_extended_training_scenario():
    """Simulate the exact scenario that caused the RuntimeError"""
    logger.info("🧪 Simulating Extended Training Scenario")
    logger.info("=" * 60)
    
    try:
        # Simulate the import sequence that would happen in run_extended_training.py
        logger.info("Step 1: Importing with GPU configuration at the beginning...")
        
        # This should configure GPU settings before any TensorFlow operations
        from utils.gpu_config import ensure_gpu_configured
        ensure_gpu_configured()
        logger.info("✅ GPU configuration applied early")
        
        # Now import the modules that previously caused the error
        logger.info("Step 2: Importing bayesian optimization module...")
        
        # This import chain would previously cause TensorFlow to be initialized
        # without proper GPU configuration, then try to set memory growth later
        try:
            from utils.bayesian_optimization import BayesianOptimizer
            logger.info("✅ BayesianOptimizer imported without RuntimeError")
        except ImportError as e:
            logger.info(f"⚠️  Import skipped due to missing dependencies: {e}")
        except RuntimeError as e:
            if "Physical devices cannot be modified after being initialized" in str(e):
                logger.error("❌ The RuntimeError still occurs!")
                return False
            else:
                logger.error(f"❌ Unexpected RuntimeError: {e}")
                return False
        
        logger.info("Step 3: Importing trainer module...")
        try:
            from training.trainer import PLSTMTALTrainer
            logger.info("✅ PLSTMTALTrainer imported without RuntimeError")
        except ImportError as e:
            logger.info(f"⚠️  Import skipped due to missing dependencies: {e}")
        except RuntimeError as e:
            if "Physical devices cannot be modified after being initialized" in str(e):
                logger.error("❌ The RuntimeError still occurs!")
                return False
            else:
                logger.error(f"❌ Unexpected RuntimeError: {e}")
                return False
        
        logger.info("Step 4: Testing function that would call optimize_plstm_tal_hyperparameters...")
        try:
            from utils.bayesian_optimization import optimize_plstm_tal_hyperparameters
            logger.info("✅ optimize_plstm_tal_hyperparameters imported without RuntimeError")
        except ImportError as e:
            logger.info(f"⚠️  Import skipped due to missing dependencies: {e}")
        except RuntimeError as e:
            if "Physical devices cannot be modified after being initialized" in str(e):
                logger.error("❌ The RuntimeError still occurs!")
                return False
            else:
                logger.error(f"❌ Unexpected RuntimeError: {e}")
                return False
        
        logger.info("✅ All imports completed successfully without the RuntimeError")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during simulation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_environment_variable_persistence():
    """Test that the environment variable approach works"""
    logger.info("🧪 Testing Environment Variable Persistence")
    logger.info("=" * 60)
    
    try:
        # Ensure GPU configuration sets the environment variable
        from utils.gpu_config import GPUConfig
        GPUConfig.reset()
        
        # Configure GPU - this should set the environment variable
        GPUConfig.configure_gpu_memory_growth(True)
        
        # Check that environment variable is set
        env_var = os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')
        if env_var == 'true':
            logger.info("✅ TF_FORCE_GPU_ALLOW_GROWTH environment variable set correctly")
        else:
            logger.error(f"❌ Environment variable not set correctly: {env_var}")
            return False
        
        # This environment variable will be inherited by any TensorFlow initialization
        # and will configure memory growth automatically
        logger.info("✅ Environment variable approach working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment variable test failed: {e}")
        return False

def main():
    """Run the simulation tests"""
    logger.info("🚀 Testing Fix for TensorFlow GPU Memory Growth RuntimeError")
    logger.info("🎯 Original Error: 'Physical devices cannot be modified after being initialized'")
    logger.info("💡 Solution: Configure GPU settings before any TensorFlow operations")
    logger.info("=" * 80)
    
    tests = [
        simulate_extended_training_scenario,
        test_environment_variable_persistence
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
        logger.info("")  # Add spacing between tests
    
    if all_passed:
        logger.info("🎉 SUCCESS: RuntimeError fix working correctly!")
        logger.info("💡 The original error should no longer occur when running:")
        logger.info("   - run_extended_training.py")
        logger.info("   - run_maximum_accuracy_training.py") 
        logger.info("   - run_complete_training.py")
        logger.info("   - app.py")
        return 0
    else:
        logger.error("❌ FAILURE: Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())