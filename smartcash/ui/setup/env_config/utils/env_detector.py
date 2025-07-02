# File: smartcash/ui/setup/env_config/utils/env_detector.py
# Deskripsi: Utility untuk deteksi informasi environment

import os
import sys
import platform
import traceback
from typing import Dict, Any

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
        result.update({
            'gpu': gpu_info,
            'gpu_available': 'No CUDA available' not in str(gpu_info),
            'is_colab': safe_get(_is_google_colab, False),
            'cpu_cores': safe_get(_get_cpu_cores, 0),
            'total_ram': safe_get(_get_total_ram, 0),
            'storage_info': safe_get(_get_storage_info, {})
        })
        
        # Get drive mount status with detailed error handling
        try:
            drive_mounted, mount_path = safe_get(_is_drive_mounted, (False, ''))
            result.update({
                'drive_mounted': drive_mounted,
                'drive_mount_path': mount_path or ''
            })
        except Exception as e:
            print(f"Warning: Error checking drive mount: {str(e)}")
            result.update({
                'drive_mounted': False,
                'drive_mount_path': '',
                'drive_mount_error': str(e)
            })
            
    except Exception as e:
        print(f"Error in detect_environment_info: {str(e)}")
        result.update({
            'status': 'error',
            'error': str(e),
            'traceback': str(traceback.format_exc())
        })
    
    return result

def _get_python_version() -> str:
    """🐍 Get Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _get_platform_info() -> str:
    """🖥️ Get platform information"""
    return f"{platform.system()} {platform.release()}"

def _get_os_info() -> Dict[str, Any]:
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
        'processor': platform.processor() or 'Unknown'
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
            return f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}"
        else:
            return "No CUDA available"
    except ImportError:
        return "PyTorch not installed"

def _is_drive_mounted() -> tuple:
    """💾 Check if Google Drive is mounted and return the mount path if available
    
    Returns:
        tuple: (is_mounted: bool, mount_path: str)
    """
    try:
        # Check if we're running in Colab first
        in_colab = _is_google_colab()
        
        # If not in Colab, no need to check for mounted drive
        if not in_colab:
            return False, ""
            
        # Check common mount paths
        mount_paths = [
            '/content/drive/MyDrive',
            '/content/gdrive/MyDrive',
            '/content/drive/My Drive'
        ]
        
        for path in mount_paths:
            try:
                if os.path.exists(path):
                    return True, path
            except Exception as e:
                # Log the error but continue checking other paths
                print(f"Error checking path {path}: {str(e)}")
                continue
                
        return False, ""
        
    except Exception as e:
        # If any error occurs, assume drive is not mounted
        print(f"Error checking drive mount status: {str(e)}")
        return False, ""

def _get_cpu_cores() -> int:
    """🖥️ Get number of CPU cores"""
    import multiprocessing
    return multiprocessing.cpu_count()

def _get_total_ram() -> str:
    """💾 Get total RAM in GB"""
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        return f"{ram_gb} GB"
    except:
        return "N/A"

def _get_storage_info() -> str:
    """💽 Get storage information"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        total_gb = total // (2**30)
        used_gb = used // (2**30)
        free_gb = free // (2**30)
        return f"{used_gb}GB / {total_gb}GB (Free: {free_gb}GB)"
    except:
        return "N/A"

def get_runtime_type() -> Dict[str, str]:
    """Get detailed runtime type information.
    
    Returns:
        Dictionary containing:
        - type: Runtime type (e.g., 'colab', 'local')
        - gpu: GPU availability ('available' or 'not available')
        - display: Formatted display string
    """
    runtime_info = {
        'type': 'local',
        'gpu': 'not available',
        'display': 'Local Environment'
    }
    
    if _is_google_colab():
        runtime_info['type'] = 'colab'
        gpu_available = "CUDA" in _get_gpu_info()
        runtime_info['gpu'] = 'available' if gpu_available else 'not available'
        runtime_info['display'] = f"Google Colab ({'GPU' if gpu_available else 'CPU'})"
    
    return runtime_info