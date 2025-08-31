"""
GPU Configuration Module for TensorFlow
Handles GPU memory growth configuration before TensorFlow initialization
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class GPUConfig:
    """Centralized GPU configuration manager"""
    
    _configured = False
    _memory_growth_enabled = False
    
    @classmethod
    def configure_gpu_memory_growth(cls, enable: bool = True) -> bool:
        """
        Configure GPU memory growth before TensorFlow initialization
        
        Args:
            enable: Whether to enable memory growth
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        if cls._configured:
            logger.debug("GPU configuration already applied")
            return cls._memory_growth_enabled
            
        try:
            # Set environment variable before TensorFlow import
            if enable:
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                logger.info("Set TF_FORCE_GPU_ALLOW_GROWTH environment variable")
            
            # Try to configure TensorFlow GPU settings
            try:
                import tensorflow as tf
                
                # Check if TensorFlow context is already initialized
                if hasattr(tf.config, 'experimental'):
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    
                    if gpus:
                        logger.info(f"Found {len(gpus)} GPU(s)")
                        
                        # Check if we can still configure memory growth
                        try:
                            for gpu in gpus:
                                # Try to get current memory growth setting
                                current_growth = tf.config.experimental.get_memory_growth(gpu)
                                
                                if current_growth == enable:
                                    logger.info(f"GPU {gpu.name} memory growth already set to {enable}")
                                else:
                                    tf.config.experimental.set_memory_growth(gpu, enable)
                                    logger.info(f"Successfully set GPU {gpu.name} memory growth to {enable}")
                                    
                        except RuntimeError as e:
                            if "Physical devices cannot be modified after being initialized" in str(e):
                                logger.warning(
                                    "TensorFlow already initialized. GPU memory growth cannot be changed. "
                                    "Using environment variable for future sessions."
                                )
                                # Memory growth will be applied via environment variable
                                cls._memory_growth_enabled = enable
                                cls._configured = True
                                return True
                            else:
                                raise e
                    else:
                        logger.info("No GPUs found, skipping GPU memory configuration")
                        
                cls._memory_growth_enabled = enable
                cls._configured = True
                logger.info(f"GPU memory growth configuration completed: {enable}")
                return True
                
            except ImportError:
                logger.info("TensorFlow not available, skipping GPU configuration")
                cls._configured = True
                return False
                
        except Exception as e:
            logger.error(f"Error configuring GPU memory growth: {e}")
            return False
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if GPU configuration has been applied"""
        return cls._configured
    
    @classmethod
    def get_memory_growth_status(cls) -> Optional[bool]:
        """Get current memory growth status"""
        if not cls._configured:
            return None
        return cls._memory_growth_enabled
    
    @classmethod
    def reset(cls):
        """Reset configuration state (for testing)"""
        cls._configured = False
        cls._memory_growth_enabled = False


def configure_tensorflow_gpu(enable_memory_growth: bool = True) -> bool:
    """
    Convenience function to configure TensorFlow GPU settings
    
    Args:
        enable_memory_growth: Whether to enable GPU memory growth
        
    Returns:
        bool: True if configuration was successful
    """
    return GPUConfig.configure_gpu_memory_growth(enable_memory_growth)


def ensure_gpu_configured():
    """
    Ensure GPU configuration is applied before any TensorFlow operations
    This should be called at the very beginning of the application
    """
    if not GPUConfig.is_configured():
        logger.info("Applying GPU configuration...")
        configure_tensorflow_gpu(enable_memory_growth=True)
    else:
        logger.debug("GPU configuration already applied")