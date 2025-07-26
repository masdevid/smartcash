#!/usr/bin/env python3
"""
File: smartcash/model/utils/memory_optimizer.py

Comprehensive memory optimization utilities for training and model building.

This module provides platform-aware memory optimization that solves common issues:
- MPS memory limitations and fragmentation on Apple Silicon
- CPU semaphore leaking with thread limiting
- CUDA memory management and mixed precision optimization
- Gradient accumulation strategies for memory-constrained environments

Features:
- Automatic device-specific memory optimization
- Emergency memory cleanup procedures
- Memory monitoring and reporting
- Platform-aware batch size and worker configuration
- Gradient accumulation optimization
"""

import os
import gc
import torch
import psutil
from typing import Dict, Any, Optional, Tuple
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class MemoryOptimizer:
    """Comprehensive memory optimization for training and model building."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory optimizer with device detection."""
        self.device = device or self._detect_optimal_device()
        self.platform_info = self._get_platform_info()
        logger.info(f"🧠 Memory optimizer initialized for {self.device}")
    
    def _detect_optimal_device(self) -> torch.device:
        """Detect the most memory-efficient device available."""
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                device = torch.device('mps')
                # Device selection logged in __init__
                return device
            except RuntimeError as e:
                logger.warning(f"⚠️ MPS cache error: {e}, continuing with MPS")
                return torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.empty_cache()
            # Device selection logged in __init__
            return device
        else:
            device = torch.device('cpu')
            # Optimize CPU threading to prevent semaphore leaking
            torch.set_num_threads(min(4, os.cpu_count() or 4))
            logger.info(f"🖥️ Using CPU with {torch.get_num_threads()} threads")
            return device
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information for optimization."""
        return {
            'device_type': self.device.type,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'cpu_count': os.cpu_count() or 4,
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'is_apple_silicon': self.device.type == 'mps',
            'is_cuda_workstation': self.device.type == 'cuda',
            'is_cpu_only': self.device.type == 'cpu'
        }
    
    def setup_memory_efficient_settings(self) -> Dict[str, Any]:
        """Configure PyTorch for memory efficiency based on platform."""
        config = {}
        
        # Platform-specific optimizations
        if self.platform_info['is_cuda_workstation']:
            # CUDA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
            config['mixed_precision'] = True
            config['compile_model'] = True
            logger.info("⚡ CUDA memory optimization enabled")
            
        elif self.device.type == 'mps':
            # MPS optimizations - aggressive memory management
            if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
                del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
            
            # Conservative MPS settings
            config['mixed_precision'] = False  # MPS doesn't support AMP
            config['compile_model'] = False   # torch.compile issues on M1
            config['memory_efficient'] = True
            logger.info("🧠 MPS conservative memory management enabled")
            
        else:
            # CPU optimizations to prevent semaphore leaking
            os.environ['OMP_NUM_THREADS'] = str(min(4, self.platform_info['cpu_count']))
            os.environ['MKL_NUM_THREADS'] = str(min(4, self.platform_info['cpu_count']))
            torch.set_num_threads(min(4, self.platform_info['cpu_count']))
            
            config['mixed_precision'] = False
            config['compile_model'] = False
            config['memory_efficient'] = True
            config['num_workers'] = 0  # Prevent semaphore leaking
            logger.info("🖥️ CPU semaphore leak prevention enabled")
        
        # Common optimizations
        torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
        config['device'] = self.device
        # Remove redundant message - already logged device-specific optimization above
        
        return config
    
    def get_optimal_batch_config(self, backbone: str = 'cspdarknet') -> Dict[str, Any]:
        """Get optimal batch size and gradient accumulation configuration."""
        # Conservative configurations to prevent OOM
        backbone_configs = {
            'efficientnet_b4': {
                'mps_batch': 1, 'cuda_batch': 4, 'cpu_batch': 1,
                'target_effective_batch': 8
            },
            'cspdarknet': {
                'mps_batch': 2, 'cuda_batch': 6, 'cpu_batch': 1,
                'target_effective_batch': 8
            }
        }
        
        config = backbone_configs.get(backbone, backbone_configs['efficientnet_b4'])
        
        # Select device-specific configuration
        if self.platform_info['is_apple_silicon']:
            batch_size = config['mps_batch']
            gradient_accumulation_steps = max(1, config['target_effective_batch'] // batch_size)
            num_workers = 0  # Always 0 for MPS to prevent issues
            pin_memory = False
            
        elif self.platform_info['is_cuda_workstation']:
            batch_size = config['cuda_batch']
            gradient_accumulation_steps = max(1, config['target_effective_batch'] // batch_size)
            num_workers = min(4, self.platform_info['cpu_count'])
            pin_memory = True
            
        else:  # CPU
            batch_size = config['cpu_batch']
            gradient_accumulation_steps = max(1, config['target_effective_batch'] // batch_size)
            num_workers = 0  # Prevent semaphore leaking
            pin_memory = False
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        result = {
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': num_workers > 0,
            'prefetch_factor': 1 if self.platform_info['is_apple_silicon'] else 2,
            'drop_last': True
        }
        
        logger.info(f"📊 Optimal batch config: {batch_size} batch × {gradient_accumulation_steps} accumulation = {effective_batch_size} effective")
        return result
    
    def cleanup_memory(self, aggressive: bool = False) -> None:
        """Force memory cleanup with platform-specific strategies."""
        # Force garbage collection
        gc.collect()
        
        if self.platform_info['is_apple_silicon']:
            try:
                # MPS memory cleanup
                cleanup_rounds = 5 if aggressive else 3
                for _ in range(cleanup_rounds):
                    torch.mps.empty_cache()
                    gc.collect()
                logger.info(f"🧹 MPS memory cleaned ({'aggressive' if aggressive else 'standard'})")
            except RuntimeError as e:
                logger.warning(f"⚠️ MPS cache warning: {e}")
                
        elif self.platform_info['is_cuda_workstation']:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if aggressive:
                torch.cuda.ipc_collect()
            logger.info(f"🧹 CUDA memory cleaned ({'aggressive' if aggressive else 'standard'})")
            
        else:
            # CPU cleanup
            if aggressive:
                for _ in range(3):
                    gc.collect()
            logger.info("🧹 CPU memory cleaned")
    
    def emergency_memory_cleanup(self) -> None:
        """Emergency memory cleanup for out-of-memory situations."""
        logger.warning("🚨 Emergency memory cleanup initiated...")
        
        # Multiple garbage collection passes
        for _ in range(5):
            gc.collect()
        
        if self.platform_info['is_apple_silicon']:
            try:
                # Aggressive MPS cleanup
                for _ in range(10):
                    torch.mps.empty_cache()
                    gc.collect()
                logger.info("🚨 Emergency MPS cleanup completed")
            except RuntimeError:
                pass
                
        elif self.platform_info['is_cuda_workstation']:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            logger.info("🚨 Emergency CUDA cleanup completed")
        
        logger.info("🚨 Emergency cleanup finished")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status for monitoring."""
        status = {
            'device': str(self.device),
            'platform': self.platform_info['device_type']
        }
        
        if self.platform_info['is_cuda_workstation']:
            try:
                status.update({
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'gpu_memory_total_gb': torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                })
            except Exception:
                status['gpu_memory'] = 'unavailable'
                
        # System memory
        memory = psutil.virtual_memory()
        status.update({
            'system_memory_used_gb': memory.used / (1024**3),
            'system_memory_total_gb': memory.total / (1024**3),
            'system_memory_percent': memory.percent
        })
        
        return status
    
    def optimize_for_training(self, model: torch.nn.Module, 
                            backbone: str = 'cspdarknet') -> Dict[str, Any]:
        """Optimize model and get training configuration."""
        # Setup memory efficient settings (check cache first)
        if not hasattr(self, '_memory_config_cache'):
            self._memory_config_cache = self.setup_memory_efficient_settings()
        memory_config = self._memory_config_cache
        
        # Get optimal batch configuration
        batch_config = self.get_optimal_batch_config(backbone)
        
        # Move model to device efficiently
        model = model.to(self.device)
        
        # Platform-specific model optimizations
        if memory_config.get('compile_model', False):
            try:
                model = torch.compile(model)
                logger.info("⚡ Model compiled for optimization")
            except Exception as e:
                logger.warning(f"⚠️ Model compilation failed: {e}")
        
        # Clean memory after model setup
        self.cleanup_memory()
        
        result = {
            'model': model,
            'device': self.device,
            'memory_config': memory_config,
            'batch_config': batch_config,
            'platform_info': self.platform_info
        }
        
        # Remove redundant message - optimization already logged during initialization
        return result
    
    def create_memory_monitor_callback(self) -> callable:
        """Create a callback for monitoring memory during training."""
        def memory_callback(phase: str, epoch: int = None, **kwargs):
            """Memory monitoring callback for training phases."""
            status = self.get_memory_status()
            
            if self.platform_info['is_apple_silicon']:
                memory_info = f"MPS - System: {status['system_memory_percent']:.1f}%"
            elif self.platform_info['is_cuda_workstation']:
                gpu_used = status.get('gpu_memory_allocated_gb', 0)
                gpu_total = status.get('gpu_memory_total_gb', 1)
                memory_info = f"GPU: {gpu_used:.1f}/{gpu_total:.1f}GB"
            else:
                memory_info = f"CPU - System: {status['system_memory_percent']:.1f}%"
            
            # Emit memory status
            logger.debug(f"📊 {phase} memory: {memory_info}")
            
            # Automatic cleanup for high memory usage
            if status['system_memory_percent'] > 85:
                logger.warning("⚠️ High memory usage detected, running cleanup")
                self.cleanup_memory(aggressive=True)
        
        return memory_callback


# Global instance for easy access
_memory_optimizer = None

def get_memory_optimizer(device: Optional[torch.device] = None) -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None or (device and _memory_optimizer.device != device):
        _memory_optimizer = MemoryOptimizer(device)
    return _memory_optimizer

def setup_memory_optimization(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Setup memory optimization and return configuration."""
    optimizer = get_memory_optimizer(device)
    return optimizer.setup_memory_efficient_settings()

def optimize_model_for_training(model: torch.nn.Module, 
                              backbone: str = 'cspdarknet',
                              device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Optimize model for memory-efficient training."""
    optimizer = get_memory_optimizer(device)
    return optimizer.optimize_for_training(model, backbone)

def cleanup_training_memory(aggressive: bool = False) -> None:
    """Clean up memory after training operations."""
    if _memory_optimizer:
        _memory_optimizer.cleanup_memory(aggressive)
    else:
        # Fallback cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass

def emergency_cleanup() -> None:
    """Emergency memory cleanup for OOM situations."""
    if _memory_optimizer:
        _memory_optimizer.emergency_memory_cleanup()
    else:
        # Fallback emergency cleanup
        for _ in range(5):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            try:
                for _ in range(5):
                    torch.mps.empty_cache()
            except RuntimeError:
                pass