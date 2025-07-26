#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/smartcash/model/training/platform_presets.py

Platform-aware training presets for optimal performance across different hardware configurations.

This module automatically detects the system environment and provides optimized training
configurations for different platforms including:
- MacBook Pro M1/M2 (Apple Silicon with MPS)
- Google Colab (CUDA GPU)
- Linux workstations (CUDA GPU)
- CPU-only environments

Features:
- Automatic hardware detection
- Memory-optimized settings per platform
- Worker count optimization using smartcash.common.worker_utils
- Mixed precision configuration
- Batch size optimization
- Platform-specific memory management
"""

import os
import platform
import torch
from typing import Dict, Any

from smartcash.common.worker_utils import get_optimal_worker_count, optimal_mixed_workers
from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer

logger = get_logger(__name__)

class PlatformPresets:
    """Platform-aware preset configurations for training."""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        logger.info(f"🖥️ Detected platform: {self.platform_info['platform_name']}")
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect current platform and hardware capabilities."""
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'python': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'cpu_count': os.cpu_count() or 1,
            'is_colab': False,
            'is_m1_mac': False,
            'is_cuda_workstation': False,
            'platform_name': 'unknown'
        }
        
        # Check for Google Colab
        try:
            import google.colab
            info['is_colab'] = True
            info['platform_name'] = 'google_colab'
        except ImportError:
            pass
        
        # Check for Apple Silicon Mac
        if info['system'] == 'Darwin' and info['machine'] == 'arm64':
            info['is_m1_mac'] = True
            info['platform_name'] = 'apple_silicon_mac'
        
        # Check for CUDA workstation
        elif info['cuda_available'] and not info['is_colab']:
            info['is_cuda_workstation'] = True
            info['platform_name'] = 'cuda_workstation'
        
        # CPU-only fallback
        elif not info['cuda_available'] and not info['mps_available']:
            info['platform_name'] = 'cpu_only'
        
        return info
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get optimized device configuration for current platform."""
        if self.platform_info['is_colab']:
            return {
                'device': 'cuda' if self.platform_info['cuda_available'] else 'cpu',
                'mixed_precision': self.platform_info['cuda_available'],
                'memory_fraction': 0.9,
                'allow_tf32': True
            }
        
        elif self.platform_info['is_m1_mac']:
            return {
                'device': 'mps' if self.platform_info['mps_available'] else 'cpu',
                'mixed_precision': False,  # MPS has mixed precision issues
                'memory_fraction': 0.7,    # Conservative for unified memory
                'allow_tf32': False
            }
        
        elif self.platform_info['is_cuda_workstation']:
            return {
                'device': 'cuda',
                'mixed_precision': True,
                'memory_fraction': 0.8,
                'allow_tf32': True
            }
        
        else:  # CPU-only
            return {
                'device': 'cpu',
                'mixed_precision': False,
                'memory_fraction': 1.0,
                'allow_tf32': False
            }
    
    def get_data_config(self, backbone: str = 'cspdarknet') -> Dict[str, Any]:
        """Get optimized data loading configuration with memory optimization."""
        device_config = self.get_device_config()
        
        # Get memory-optimized batch configuration
        try:
            memory_optimizer = get_memory_optimizer()
            batch_config = memory_optimizer.get_optimal_batch_config(backbone)
            
            # Use memory optimizer configuration as base
            config = {
                'batch_size': batch_config['batch_size'],
                'num_workers': batch_config['num_workers'],
                'pin_memory': batch_config['pin_memory'],
                'persistent_workers': batch_config['persistent_workers'],
                'prefetch_factor': batch_config['prefetch_factor'],
                'drop_last': batch_config['drop_last']
            }
        except Exception as e:
            logger.warning(f"⚠️ Memory optimizer unavailable, using fallback config: {e}")
            # Fallback configuration
            config = {
                'num_workers': optimal_mixed_workers(),  # Mixed I/O and CPU operations
                'pin_memory': device_config['device'] == 'cuda',  # Only beneficial for CUDA
                'persistent_workers': True,
                'prefetch_factor': 2,
                'drop_last': True
            }
        
        # Platform-specific optimizations
        if self.platform_info['is_colab']:
            # Colab has good I/O but limited memory
            config.update({
                'batch_size': 16 if device_config['device'] == 'cuda' else 8,
                'num_workers': min(4, config['num_workers']),  # Limit for Colab
                'prefetch_factor': 1  # Reduce memory usage
            })
        
        elif self.platform_info['is_m1_mac']:
            # M1 Mac: unified memory, be conservative
            base_batch = 6 if backbone == 'cspdarknet' else 4  # EfficientNet needs less
            config.update({
                'batch_size': base_batch,
                'num_workers': min(6, config['num_workers']),  # Good CPU, limit for memory
                'pin_memory': False,  # Not beneficial for MPS
                'persistent_workers': config['num_workers'] > 0,
                'prefetch_factor': 1  # Conservative memory usage
            })
            
            # Set MPS memory environment
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
            os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'native'
        
        elif self.platform_info['is_cuda_workstation']:
            # CUDA workstation: maximize throughput
            config.update({
                'batch_size': 24 if backbone == 'cspdarknet' else 16,
                'num_workers': config['num_workers'],  # Use full workers
                'prefetch_factor': 2
            })
        
        else:  # CPU-only
            # CPU: small batches, more workers for I/O
            config.update({
                'batch_size': 4,
                'num_workers': get_optimal_worker_count('io'),  # I/O bound for CPU
                'pin_memory': False,
                'persistent_workers': False  # Less memory overhead
            })
        
        return config
    
    def get_training_config(self, backbone: str = 'cspdarknet') -> Dict[str, Any]:
        """Get complete training configuration optimized for platform."""
        device_config = self.get_device_config()
        data_config = self.get_data_config(backbone)
        
        # Base training configuration
        config = {
            'mixed_precision': device_config['mixed_precision'],
            'gradient_clip': 10.0,
            'weight_decay': 0.0005,
            'optimizer': 'adamw',  # Generally better than adam
            'scheduler': 'cosine',  # String value expected by OptimizerFactory
            'scheduler_config': {  # Separate config for scheduler details
                'type': 'cosine',
                'eta_min': 1e-6,
                'warmup_epochs': 0  # Will be set per phase
            },
            'early_stopping': {  # Add early stopping configuration
                'enabled': True,
                'patience': 10,
                'metric': 'val_map50',
                'mode': 'max',
                'min_delta': 0.002
            }
        }
        
        # Platform-specific optimizations
        if self.platform_info['is_m1_mac']:
            # M1 Mac optimizations
            config.update({
                'mixed_precision': False,  # Avoid MPS issues
                'gradient_clip': 5.0,      # More conservative
                'compile_model': False,    # torch.compile issues on M1
                'memory_efficient': True
            })
        
        elif self.platform_info['is_colab']:
            # Colab optimizations
            config.update({
                'gradient_clip': 10.0,
                'compile_model': True,     # Good for Colab's newer PyTorch
                'memory_efficient': False  # Colab has decent memory
            })
        
        elif self.platform_info['is_cuda_workstation']:
            # Workstation optimizations
            config.update({
                'gradient_clip': 15.0,
                'compile_model': True,     # Use torch.compile for speed
                'memory_efficient': False
            })
        
        else:  # CPU-only
            config.update({
                'mixed_precision': False,
                'gradient_clip': 5.0,
                'compile_model': False,    # torch.compile primarily for GPU
                'memory_efficient': True,
                'optimizer': 'sgd'         # Often better on CPU
            })
        
        return config
    
    def get_model_config(self, backbone: str = 'cspdarknet') -> Dict[str, Any]:
        """Get model configuration optimized for platform."""
        device_config = self.get_device_config()
        
        config = {
            'backbone': backbone,
            'pretrained': False,           # Use non-pretrained models for training from scratch
            'feature_optimization': True,  # Always enable
            'layer_mode': 'multi',
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
            'multi_layer_heads': True,
            'num_classes': {
                'layer_1': 7,  # Banknote denominations
                'layer_2': 7,  # Denomination features  
                'layer_3': 3   # Common features
            },
            'img_size': 640
        }
        
        # Platform-specific model optimizations
        if self.platform_info['is_m1_mac']:
            # M1 Mac: smaller input size for memory efficiency
            if backbone == 'efficientnet_b4':
                config['img_size'] = 512  # Reduce for memory
        
        return config
    
    def get_phase_config(self, phase_1_epochs: int = 1, phase_2_epochs: int = 1) -> Dict[str, Any]:
        """Get phase-specific training configuration."""
        return {
            'phase_1': {
                'description': 'Train detection heads with frozen backbone',
                'epochs': phase_1_epochs,
                'freeze_backbone': True,
                'learning_rates': {
                    'backbone': 1e-5,  # Very low for frozen backbone
                    'head': 1e-3       # Higher for training heads
                },
                'warmup_epochs': 0  # No warmup for short training
            },
            'phase_2': {
                'description': 'Fine-tune entire model',
                'epochs': phase_2_epochs,
                'freeze_backbone': False,
                'learning_rates': {
                    'backbone': 5e-6,  # Very conservative for backbone
                    'head': 1e-4       # Moderate for heads
                },
                'warmup_epochs': 0  # No warmup for short training
            }
        }
    
    def get_full_config(self, backbone: str = 'cspdarknet', 
                       phase_1_epochs: int = 1, phase_2_epochs: int = 1) -> Dict[str, Any]:
        """Get complete platform-optimized configuration."""
        device_config = self.get_device_config()
        data_config = self.get_data_config(backbone)
        training_config = self.get_training_config(backbone)
        model_config = self.get_model_config(backbone)
        phase_config = self.get_phase_config(phase_1_epochs, phase_2_epochs)
        
        return {
            'device': device_config,
            'data': data_config,
            'training': training_config,
            'model': model_config,
            'training_phases': phase_config,
            'platform_info': self.platform_info,
            'paths': {
                'checkpoints': 'data/checkpoints',
                'visualization': 'data/visualization',
                'logs': 'data/logs'
            }
        }
    
    def setup_platform_optimizations(self):
        """Apply platform-specific PyTorch optimizations with memory optimization."""
        try:
            # Use memory optimizer for comprehensive optimization
            memory_optimizer = get_memory_optimizer()
            memory_config = memory_optimizer.setup_memory_efficient_settings()
            logger.info(f"✅ Memory optimization applied: {memory_config.get('device', 'unknown')}")
            return memory_config
        except Exception as e:
            logger.warning(f"⚠️ Memory optimizer unavailable, using fallback optimizations: {e}")
            
            # Fallback platform optimizations
            if self.platform_info['is_cuda_workstation'] or self.platform_info['is_colab']:
                # CUDA optimizations
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.cuda.empty_cache()
            
            elif self.platform_info['is_m1_mac']:
                # MPS optimizations
                if self.platform_info['mps_available']:
                    torch.mps.empty_cache()
            
            # General optimizations
            torch.set_float32_matmul_precision('medium')  # Allow TensorFloat-32
            return {'fallback': True}


# Global instance for easy access
_platform_presets = None

def get_platform_presets() -> PlatformPresets:
    """Get global platform presets instance."""
    global _platform_presets
    if _platform_presets is None:
        _platform_presets = PlatformPresets()
    return _platform_presets

def get_platform_config(backbone: str = 'cspdarknet', 
                       phase_1_epochs: int = 1, 
                       phase_2_epochs: int = 1) -> Dict[str, Any]:
    """Get platform-optimized configuration."""
    presets = get_platform_presets()
    return presets.get_full_config(backbone, phase_1_epochs, phase_2_epochs)

def setup_platform_optimizations():
    """Setup platform-specific optimizations with memory optimization."""
    presets = get_platform_presets()
    return presets.setup_platform_optimizations()