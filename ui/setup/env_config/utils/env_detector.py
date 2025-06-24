"""
File: smartcash/ui/setup/env_config/utils/env_detector.py
Deskripsi: Utility untuk deteksi informasi environment
"""

import os
import sys
import platform
from typing import Dict, Any

def detect_environment_info() -> Dict[str, Any]:
    """🔍 Deteksi informasi environment lengkap"""
    return {
        'python_version': _get_python_version(),
        'platform': _get_platform_info(),
        'is_colab': _is_google_colab(),
        'gpu_info': _get_gpu_info(),
        'drive_mounted': _is_drive_mounted()[0],
        'drive_mount_path': _is_drive_mounted()[1],
        'cpu_cores': _get_cpu_cores(),
        'total_ram': _get_total_ram(),
        'storage_info': _get_storage_info()
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

def _is_drive_mounted() -> tuple[bool, str]:
    """💾 Check if Google Drive is mounted and return the mount path if available
    
    Returns:
        tuple: (is_mounted, mount_path) where mount_path is the path where drive is mounted,
              or an empty string if not mounted
    """
    mount_paths = [
        '/content/drive/MyDrive',
        '/content/gdrive/MyDrive',
        '/content/drive/My Drive'
    ]
    for path in mount_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return True, path
    return False, ''

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