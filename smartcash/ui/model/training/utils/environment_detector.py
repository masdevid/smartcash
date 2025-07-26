"""
File: smartcash/ui/model/training/utils/environment_detector.py
Description: Environment detection utilities for training configuration using mixin pattern.
"""

import os
import torch
from typing import Dict, Any, Tuple
from smartcash.ui.core.mixins.environment_mixin import EnvironmentMixin
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)


class TrainingEnvironmentDetector(EnvironmentMixin):
    """Training-specific environment detector using EnvironmentMixin."""
    
    def __init__(self):
        super().__init__()
        self.logger = logger


    def detect_training_environment(self) -> Dict[str, Any]:
        """
        Detect environment with training-specific enhancements.
        
        Returns:
            Dictionary containing training-specific environment information
        """
        # Use base environment detection from mixin
        base_env = self.get_environment_info()
        
        # Add PyTorch-specific detection
        training_env = {
            'is_colab': self.is_colab,
            'is_local': not self.is_colab,
            'has_gpu': False,
            'has_mps': False,
            'cuda_available': False,
            'mps_available': False,
            'device_count': 0,
            'device_names': [],
            'platform': base_env.get('environment_type', 'unknown'),
            'force_cpu_recommended': False,
            'recommended_batch_size': None,
            'recommended_epochs': None
        }
        
        # Detect PyTorch devices
        try:
            if torch.cuda.is_available():
                training_env['cuda_available'] = True
                training_env['has_gpu'] = True
                training_env['device_count'] = torch.cuda.device_count()
                training_env['device_names'] = [
                    torch.cuda.get_device_name(i) for i in range(training_env['device_count'])
                ]
                self.logger.info(f"🎮 CUDA GPU(s) detected: {training_env['device_names']}")
            
            # Check MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                training_env['mps_available'] = True
                training_env['has_gpu'] = True
                if not training_env['cuda_available']:  # Only set if no CUDA
                    training_env['device_count'] = 1
                    training_env['device_names'] = ['Apple MPS']
                self.logger.info("🍎 Apple MPS (Metal Performance Shaders) detected")
        except Exception as e:
            self.logger.error(f"❌ Error detecting PyTorch devices: {e}")
        
        try:
            # Determine if CPU training should be forced
            if training_env['is_colab'] and not training_env['cuda_available']:
                training_env['force_cpu_recommended'] = True
                self.logger.warning("⚠️ Colab environment without GPU detected - forcing CPU training")
            
            # Additional Colab-specific checks
            if training_env['is_colab']:
                # Check if GPU runtime is enabled but no GPU is available
                gpu_runtime_expected = os.environ.get('COLAB_GPU', '0') == '1'
                if gpu_runtime_expected and not training_env['cuda_available']:
                    self.logger.warning("⚠️ Colab GPU runtime expected but no GPU available")
                    training_env['force_cpu_recommended'] = True
        
        except Exception as e:
            self.logger.error(f"❌ Error during training environment detection: {e}")
            # Safe fallback
            training_env['force_cpu_recommended'] = True
        
        return training_env

    def should_force_cpu_training(self, environment_info: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Determine if CPU training should be forced based on environment.
        
        Args:
            environment_info: Optional pre-detected environment info
            
        Returns:
            Tuple of (should_force_cpu, reason)
        """
        if environment_info is None:
            environment_info = self.detect_training_environment()
        
        # Force CPU in Colab without GPU
        if environment_info['is_colab'] and not environment_info['has_gpu']:
            return True, "Colab environment without GPU detected"
        
        # Force CPU if no compute devices available
        if not environment_info['has_gpu'] and not environment_info['cuda_available'] and not environment_info['mps_available']:
            return True, "No GPU or MPS devices available"
        
        return False, "GPU/MPS available for training"

    def get_recommended_training_config(self, environment_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get recommended training configuration based on environment.
        
        Args:
            environment_info: Optional pre-detected environment info
            
        Returns:
            Dictionary containing recommended configuration updates
        """
        if environment_info is None:
            environment_info = self.detect_training_environment()
        
        config_updates = {}
        
        # Handle CPU forcing
        force_cpu, reason = self.should_force_cpu_training(environment_info)
        if force_cpu:
            config_updates['force_cpu'] = True
            self.logger.info(f"🔧 Forcing CPU training: {reason}")
        
        # Adjust batch size based on environment
        if environment_info['is_colab']:
            if environment_info['has_gpu']:
                # Colab with GPU - moderate batch size
                config_updates['batch_size'] = 8
            else:
                # Colab without GPU - small batch size for CPU
                config_updates['batch_size'] = 2
                config_updates['force_cpu'] = True
        elif environment_info['mps_available']:
            # Apple Silicon - conservative batch size due to unified memory
            config_updates['batch_size'] = 4
        elif environment_info['cuda_available']:
            # Local CUDA - can handle larger batches
            config_updates['batch_size'] = 16
        else:
            # CPU only - small batch size
            config_updates['batch_size'] = 2
            config_updates['force_cpu'] = True
        
        # Adjust epochs for faster iteration in resource-constrained environments
        if environment_info['is_colab'] or force_cpu:
            config_updates['phase_1_epochs'] = 1
            config_updates['phase_2_epochs'] = 1
            self.logger.info("🔧 Reducing epochs for resource-constrained environment")
        
        return config_updates


# Convenience functions for backwards compatibility
def detect_environment() -> Dict[str, Any]:
    """
    Detect the current environment and available compute resources.
    
    Returns:
        Dictionary containing environment information
    """
    detector = TrainingEnvironmentDetector()
    return detector.detect_training_environment()


def should_force_cpu_training(environment_info: Dict[str, Any] = None) -> Tuple[bool, str]:
    """
    Determine if CPU training should be forced based on environment.
    
    Args:
        environment_info: Optional pre-detected environment info
        
    Returns:
        Tuple of (should_force_cpu, reason)
    """
    detector = TrainingEnvironmentDetector()
    return detector.should_force_cpu_training(environment_info)


def get_recommended_training_config(environment_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get recommended training configuration based on environment.
    
    Args:
        environment_info: Optional pre-detected environment info
        
    Returns:
        Dictionary containing recommended configuration updates
    """
    detector = TrainingEnvironmentDetector()
    return detector.get_recommended_training_config(environment_info)


def log_environment_summary(environment_info: Dict[str, Any] = None) -> None:
    """
    Log a summary of the detected environment.
    
    Args:
        environment_info: Optional pre-detected environment info
    """
    if environment_info is None:
        environment_info = detect_environment()
    
    logger.info("🖥️ Environment Summary:")
    logger.info(f"   Platform: {environment_info['platform']}")
    logger.info(f"   Is Colab: {environment_info['is_colab']}")
    logger.info(f"   CUDA Available: {environment_info['cuda_available']}")
    logger.info(f"   MPS Available: {environment_info['mps_available']}")
    logger.info(f"   Device Count: {environment_info['device_count']}")
    if environment_info['device_names']:
        logger.info(f"   Devices: {', '.join(environment_info['device_names'])}")
    logger.info(f"   Force CPU Recommended: {environment_info['force_cpu_recommended']}")