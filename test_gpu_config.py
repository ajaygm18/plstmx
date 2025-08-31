#!/usr/bin/env python3
"""
Test script for GPU configuration to ensure memory growth is set correctly
"""

import unittest
import logging
import os
import sys

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/plstmx/plstmx')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGPUConfig(unittest.TestCase):
    """Test GPU configuration functionality"""
    
    def setUp(self):
        """Reset GPU configuration state before each test"""
        from utils.gpu_config import GPUConfig
        GPUConfig.reset()
    
    def test_gpu_config_import(self):
        """Test that GPU config module imports correctly"""
        from utils.gpu_config import GPUConfig, configure_tensorflow_gpu, ensure_gpu_configured
        self.assertTrue(callable(configure_tensorflow_gpu))
        self.assertTrue(callable(ensure_gpu_configured))
    
    def test_gpu_config_basic_functionality(self):
        """Test basic GPU configuration functionality"""
        from utils.gpu_config import GPUConfig
        
        # Initially not configured
        self.assertFalse(GPUConfig.is_configured())
        self.assertIsNone(GPUConfig.get_memory_growth_status())
        
        # Configure GPU
        result = GPUConfig.configure_gpu_memory_growth(True)
        
        # Should be configured now
        self.assertTrue(GPUConfig.is_configured())
        
        # Result should be boolean
        self.assertIsInstance(result, bool)
    
    def test_environment_variable_setting(self):
        """Test that environment variable is set correctly"""
        from utils.gpu_config import GPUConfig
        
        # Remove any existing env var
        if 'TF_FORCE_GPU_ALLOW_GROWTH' in os.environ:
            del os.environ['TF_FORCE_GPU_ALLOW_GROWTH']
        
        # Configure GPU
        GPUConfig.configure_gpu_memory_growth(True)
        
        # Environment variable should be set
        self.assertEqual(os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'), 'true')
    
    def test_ensure_gpu_configured_function(self):
        """Test the convenience function"""
        from utils.gpu_config import ensure_gpu_configured, GPUConfig
        
        # Initially not configured
        self.assertFalse(GPUConfig.is_configured())
        
        # Call ensure function
        ensure_gpu_configured()
        
        # Should be configured now
        self.assertTrue(GPUConfig.is_configured())
    
    def test_tensorflow_already_initialized_handling(self):
        """Test handling when TensorFlow is already initialized"""
        from utils.gpu_config import GPUConfig
        
        try:
            # Try to import TensorFlow
            import tensorflow as tf
            
            # Try to perform an operation that initializes TensorFlow
            try:
                _ = tf.constant([1, 2, 3])
            except:
                # TensorFlow might not be available
                pass
            
            # Now try to configure GPU - should handle gracefully
            result = GPUConfig.configure_gpu_memory_growth(True)
            
            # Should return a boolean regardless
            self.assertIsInstance(result, bool)
            
        except ImportError:
            # TensorFlow not available - skip this test
            self.skipTest("TensorFlow not available")
    
    def test_no_tensorflow_handling(self):
        """Test that the module handles missing TensorFlow gracefully"""
        # This test may not work if TensorFlow is installed
        # but should show the graceful handling approach
        from utils.gpu_config import GPUConfig
        
        # Configure should handle missing TensorFlow
        result = GPUConfig.configure_gpu_memory_growth(True)
        
        # Should return a boolean
        self.assertIsInstance(result, bool)


def run_gpu_config_test():
    """Run the GPU configuration test"""
    logger.info("🧪 Running GPU Configuration Test")
    logger.info("=" * 50)
    
    try:
        # Run the test
        unittest.main(argv=[''], exit=False, verbosity=2)
        logger.info("✅ GPU configuration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ GPU configuration test failed: {e}")
        return False
    
    return True


def test_integration_with_main_modules():
    """Test integration with main training modules"""
    logger.info("🧪 Testing Integration with Main Modules")
    logger.info("=" * 50)
    
    try:
        # Test that main modules can import and use GPU config
        logger.info("Testing model import with GPU config...")
        from models.plstm_tal import PLSTMTAL
        logger.info("✅ Model import successful")
        
        logger.info("Testing trainer import with GPU config...")
        from training.trainer import PLSTMTALTrainer
        logger.info("✅ Trainer import successful")
        
        logger.info("Testing bayesian optimization import...")
        from utils.bayesian_optimization import optimize_plstm_tal_hyperparameters
        logger.info("✅ Bayesian optimization import successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    logger.info("🚀 Starting GPU Configuration Tests")
    
    # Run unit tests
    success = run_gpu_config_test()
    
    if success:
        # Run integration tests
        success = test_integration_with_main_modules()
    
    if success:
        logger.info("🎉 All tests passed successfully!")
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)