"""
File: smartcash/ui/setup/colab/utils/env_detector.py
Deskripsi: Utility untuk deteksi informasi environment

Fungsi-fungsi ini digunakan untuk mendeteksi informasi lingkungan seperti:
- Versi Python
- Informasi sistem operasi
- Ketersediaan GPU
- Informasi penyimpanan
- Tipe runtime (Colab/local)
"""

import os
import sys
import platform
import traceback
from typing import Dict, Any, Tuple

def detect_environment_info() -> Dict[str, Any]:
    """Detect and return comprehensive environment information.
    
    Returns:
        Dictionary containing detailed environment information including:
        - python_version: Current Python version
        - os: Operating system information
        - gpu: GPU information if available
        - storage_info: Disk storage information
        - runtime: Runtime information (type, GPU availability)
        - is_colab: Boolean indicating if running in Google Colab
        - cpu_cores: Number of CPU cores
        - total_ram: Total RAM in bytes
        - drive_mounted: Boolean indicating if drive is mounted
        - drive_mount_path: Path where drive is mounted
    """
    # Helper function to safely get values
    def safe_get(func, default=None, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Error in {func.__name__}: {str(e)}")
            return default
    
    # Initialize result with basic info
    result = {
        'python_version': safe_get(_get_python_version, 'Unknown'),
        'os': safe_get(_get_os_info, 'Unknown'),
        'status': 'success'
    }
    
    # Get runtime information
    runtime_info = safe_get(get_runtime_type, {})
    if runtime_info:
        result['runtime'] = runtime_info
        result['runtime_display'] = runtime_info.get('display', 'Unknown')
    
    # Get additional info with error handling
    try:
        # Get GPU info
        gpu_info = safe_get(_get_gpu_info, 'Unknown')
        if gpu_info != 'Unknown':
            result['gpu'] = gpu_info
        
        # Get storage info
        storage_info = safe_get(_get_storage_info, {})
        if storage_info:
            result['storage_info'] = storage_info
        
        # Get CPU and RAM info
        cpu_cores = safe_get(_get_cpu_cores, 'Unknown')
        if cpu_cores != 'Unknown':
            result['cpu_cores'] = cpu_cores
        
        total_ram = safe_get(_get_total_ram, 'Unknown')
        if total_ram != 'Unknown':
            result['total_ram'] = total_ram
        
        # Check Google Drive mount status
        is_drive_mounted, mount_path = safe_get(_is_drive_mounted, (False, ''))
        result['drive_mounted'] = is_drive_mounted
        if is_drive_mounted and mount_path:
            result['drive_mount_path'] = mount_path
        
        # Set Colab flag
        result['is_colab'] = safe_get(_is_google_colab, False)
        
    except Exception as e:
        result['status'] = f'error: {str(e)}'
        traceback.print_exc()
    
    return result

def _get_python_version() -> str:
    """🐍 Get Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _get_platform_info() -> str:
    """🖥️ Get platform information"""
    return f"{platform.system()} {platform.release()} ({platform.machine()})"

def _get_os_info() -> Dict[str, str]:
    """Get detailed OS information.
    
    Returns:
        Dictionary containing OS information including:
        - system: OS name (e.g., 'Linux', 'Windows', 'Darwin')
        - release: OS release version
        - version: OS version
        - machine: Machine type (e.g., 'x86_64')
        - processor: Processor information
    """
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor() or 'Unknown',
        'platform': platform.platform(),
        'node': platform.node()
    }

def _is_google_colab() -> bool:
    """🔍 Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def _get_gpu_info() -> str:
    """🎮 Get GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return 'No GPU available'
    except ImportError:
        return 'PyTorch not available'

def _is_drive_mounted() -> Tuple[bool, str]:
    """💾 Check if Google Drive is mounted and return the mount path if available
    
    Returns:
        tuple: (is_mounted: bool, mount_path: str)
    """
    try:
        # Check if running in Colab
        if not _is_google_colab():
            return False, ''
            
        from google.colab import drive
        import os
        
        # Default mount path in Colab
        mount_path = '/content/drive'
        
        # Check if already mounted
        if os.path.ismount(mount_path):
            return True, mount_path
            
        # Try to mount if not already mounted
        try:
            drive.mount(mount_path)
            return True, mount_path
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            return False, ''
            
    except Exception as e:
        print(f"Error checking Google Drive mount: {e}")
        return False, ''

def _get_cpu_cores() -> int:
    """🖥️ Get number of CPU cores"""
    import multiprocessing
    return multiprocessing.cpu_count()

def _get_total_ram() -> int:
    """💾 Get total RAM in GB"""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        return 0

def _get_storage_info() -> Dict[str, Any]:
    """💽 Get storage information"""
    try:
        import psutil
        return {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent
        }
    except ImportError:
        return {}

def get_runtime_type() -> Dict[str, str]:
    """Get detailed runtime type information.
    
    Returns:
        Dictionary containing:
        - type: Runtime type (e.g., 'colab', 'local')
        - gpu: GPU availability ('available' or 'not available')
        - display: Formatted display string
    """
    try:
        is_colab = _is_google_colab()
        has_gpu = _get_gpu_info() != 'No GPU available'
        
        runtime_type = 'colab' if is_colab else 'local'
        gpu_status = 'available' if has_gpu else 'not available'
        
        return {
            'type': runtime_type,
            'gpu': gpu_status,
            'display': f"{runtime_type.capitalize()} ({gpu_status} GPU)"
        }
    except Exception as e:
        return {
            'type': 'unknown',
            'gpu': 'unknown',
            'display': f'Unknown ({str(e)})'
        }
