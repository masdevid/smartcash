# File: smartcash/ui/setup/env_config/utils/env_detector.py
# Deskripsi: Utility untuk deteksi informasi environment

import os
import sys
import platform
from typing import Dict, Any

def detect_environment_info() -> Dict[str, Any]:
    """🔍 Deteksi informasi environment lengkap"""
    try:
        # Get drive mount status once to avoid multiple checks
        drive_mounted, mount_path = _is_drive_mounted()
        
        return {
            'python_version': _get_python_version(),
            'platform': _get_platform_info(),
            'is_colab': _is_google_colab(),
            'gpu_info': _get_gpu_info(),
            'drive_mounted': drive_mounted,
            'drive_mount_path': mount_path,
            'cpu_cores': _get_cpu_cores(),
            'total_ram': _get_total_ram(),
            'storage_info': _get_storage_info(),
            'status': 'success'
        }
    except Exception as e:
        # Return a minimal set of information with error status
        return {
            'python_version': _get_python_version(),
            'platform': _get_platform_info(),
            'is_colab': _is_google_colab(),
            'drive_mounted': False,
            'drive_mount_path': None,
            'status': 'error',
            'error': str(e)
        }

def _get_python_version() -> str:
    """🐍 Get Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _get_platform_info() -> str:
    """🖥️ Get platform information"""
    return f"{platform.system()} {platform.release()}"

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

def _get_runtime_type() -> str:
    """⚡ Get runtime type information"""
    if _is_google_colab():
        gpu_available = "GPU" if "CUDA" in _get_gpu_info() else "CPU"
        return f"Colab {gpu_available}"
    return "Local"