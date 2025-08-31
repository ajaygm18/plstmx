#!/usr/bin/env python3
"""
Simple test for GPU configuration without dependencies
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/plstmx/plstmx')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_config_basic():
    """Test basic GPU configuration functionality"""
    logger.info("🧪 Testing GPU Configuration Basic Functionality")
    
    try:
        from utils.gpu_config import GPUConfig, configure_tensorflow_gpu, ensure_gpu_configured
        
        # Reset state
        GPUConfig.reset()
        
        # Test initial state
        assert not GPUConfig.is_configured(), "Should not be configured initially"
        assert GPUConfig.get_memory_growth_status() is None, "Should return None initially"
        
        # Test configuration
        result = configure_tensorflow_gpu(True)
        assert isinstance(result, bool), "Should return boolean"
        assert GPUConfig.is_configured(), "Should be configured after call"
        
        # Test environment variable
        assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') == 'true', "Environment variable should be set"
        
        # Test ensure function
        GPUConfig.reset()
        ensure_gpu_configured()
        assert GPUConfig.is_configured(), "Should be configured after ensure call"
        
        logger.info("✅ All basic GPU configuration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU configuration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_error_scenario_simulation():
    """Test error handling scenarios"""
    logger.info("🧪 Testing Error Scenarios")
    
    try:
        from utils.gpu_config import GPUConfig
        
        # Test that multiple calls are handled gracefully
        GPUConfig.reset()
        result1 = GPUConfig.configure_gpu_memory_growth(True)
        result2 = GPUConfig.configure_gpu_memory_growth(True)
        
        assert isinstance(result1, bool), "First call should return boolean"
        assert isinstance(result2, bool), "Second call should return boolean"
        
        logger.info("✅ Error scenario tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error scenario test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Simple GPU Configuration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_gpu_config_basic,
        test_error_scenario_simulation
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All tests passed successfully!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())