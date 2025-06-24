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
        'drive_mounted': _is_drive_mounted(),
        'runtime_type': _get_runtime_type()
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

def _is_drive_mounted() -> bool:
    """💾 Check if Google Drive is mounted"""
    return os.path.exists('/content/drive/MyDrive')

def _get_runtime_type() -> str:
    """⚡ Get runtime type information"""
    if _is_google_colab():
        gpu_available = "GPU" if "CUDA" in _get_gpu_info() else "CPU"
        return f"Colab {gpu_available}"
    return "Local"