"""
File: smartcash/model/utils/device_utils.py
Deskripsi: Utilities untuk device management dan CUDA setup
"""

import torch
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

def setup_device(device_config: Dict[str, Any]) -> torch.device:
    """🎮 Setup device berdasarkan konfigurasi dengan auto-detection"""
    logger = get_logger("model.device")
    
    # Check if device is explicitly set (force_cpu mode)
    explicit_device = device_config.get('device')
    if explicit_device and explicit_device != 'auto':
        device = torch.device(explicit_device)
        logger.info(f"🔧 Device explicitly set to: {device}")
        return device
    
    auto_detect = device_config.get('auto_detect', True)
    preferred = device_config.get('preferred', 'cuda')
    
    if auto_detect:
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available() and preferred == 'cuda':
            device = torch.device('cuda')
            logger.info(f"🎮 CUDA detected: {torch.cuda.get_device_name()}")
            
            # Setup CUDA optimizations
            if device_config.get('mixed_precision', True):
                logger.info("⚡ Mixed precision enabled")
            
            # CUDA memory management
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("🍎 MPS (Metal Performance Shaders) detected - optimized for M1/M2")
            
            # MPS doesn't support mixed precision yet
            if device_config.get('mixed_precision', False):
                logger.warning("⚠️ Mixed precision not supported on MPS, disabled")
            
        else:
            device = torch.device('cpu')
            logger.info("💻 Using CPU")
            
            if preferred in ['cuda', 'mps']:
                logger.warning(f"⚠️ {preferred.upper()} preferred but not available, falling back to CPU")
    else:
        # Manual device setting
        if preferred == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("🍎 MPS manually set - optimized for M1/M2")
        else:
            device = torch.device(preferred)
            logger.info(f"🔧 Device manually set to: {device}")
    
    return device

def get_device_info() -> Dict[str, Any]:
    """📊 Get comprehensive device information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        info.update({
            'current_device': current_device,
            'gpu_name': torch.cuda.get_device_name(current_device),
            'gpu_memory_gb': torch.cuda.get_device_properties(current_device).total_memory / (1024**3),
            'gpu_compute_capability': torch.cuda.get_device_properties(current_device).major,
        })
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info.update({
            'device_type': 'mps',
            'device_name': 'Apple Silicon GPU (MPS)',
            'optimization': 'Metal Performance Shaders'
        })
    
    return info

def optimize_cuda_settings() -> None:
    """⚡ Optimize CUDA settings untuk performance"""
    if torch.cuda.is_available():
        # Enable benchmark untuk consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Disable deterministic untuk better performance
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction jika diperlukan
        # torch.cuda.set_per_process_memory_fraction(0.8)

def clear_cuda_cache() -> None:
    """🧹 Clear CUDA cache untuk free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage(device: torch.device) -> Dict[str, float]:
    """📊 Get current memory usage"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        cached = torch.cuda.memory_reserved(device) / (1024**2)  # MB
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
        
        return {
            'allocated_mb': allocated,
            'cached_mb': cached,
            'total_mb': total,
            'free_mb': total - allocated,
            'utilization_pct': (allocated / total) * 100
        }
    else:
        return {'message': 'Memory tracking not available for CPU'}

def set_mixed_precision(enabled: bool = True) -> None:
    """⚡ Setup mixed precision training"""
    if enabled and torch.cuda.is_available():
        # Enable automatic mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

# Convenience functions
def is_cuda_available() -> bool:
    """🎮 Quick check CUDA availability"""
    return torch.cuda.is_available()

def get_optimal_batch_size(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> int:
    """📊 Estimate optimal batch size untuk model"""
    if device.type == 'cpu':
        return 8  # Conservative untuk CPU
    
    # Estimate berdasarkan GPU memory
    memory_info = get_memory_usage(device)
    available_memory = memory_info.get('free_mb', 1000)
    
    # Rule of thumb: start dengan batch size yang conservative
    if available_memory > 8000:  # >8GB free
        return 16
    elif available_memory > 4000:  # >4GB free
        return 8
    else:
        return 4

def model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """📱 Transfer model ke device dengan logging"""
    logger = get_logger("model.device")
    
    try:
        model = model.to(device)
        
        # Log transfer info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"📱 Model transferred to {device} | Params: {param_count:,}")
        
        # Log memory usage jika CUDA
        if device.type == 'cuda':
            memory_info = get_memory_usage(device)
            logger.info(f"🎮 GPU Memory: {memory_info['allocated_mb']:.1f}MB allocated")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to transfer model to {device}: {str(e)}")
        raise RuntimeError(f"Model transfer failed: {str(e)}")

def tensor_to_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
    """📱 Transfer tensor ke device dengan optimization"""
    return tensor.to(device, non_blocking=non_blocking)

# Device context manager
class DeviceContext:
    """🎯 Context manager untuk device operations"""
    
    def __init__(self, device: torch.device, clear_cache: bool = False):
        self.device = device
        self.clear_cache = clear_cache
        self.original_device = None
    
    def __enter__(self):
        if self.device.type == 'cuda':
            self.original_device = torch.cuda.current_device()
            torch.cuda.set_device(self.device)
            
            if self.clear_cache:
                clear_cuda_cache()
        
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda' and self.original_device is not None:
            torch.cuda.set_device(self.original_device)
            
            if self.clear_cache:
                clear_cuda_cache()

# Factory function
def create_device_context(device: torch.device, clear_cache: bool = False) -> DeviceContext:
    """🏭 Create device context manager"""
    return DeviceContext(device, clear_cache)