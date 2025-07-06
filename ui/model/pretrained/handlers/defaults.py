"""
File: smartcash/ui/pretrained/handlers/defaults.py
Deskripsi: Default configuration dan constants untuk pretrained models
"""

from typing import Dict, Any

# Default configuration untuk pretrained models
DEFAULT_CONFIG = {
    'pretrained_models': {
        'models_dir': '/data/pretrained',
        'pretrained_type': 'yolov5s',  # Hardcoded to yolov5s
        'sync_drive': False  # Disabled by default
    }
}

# Default model download URLs
DEFAULT_MODEL_URLS = {
    'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
    'efficientnet': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b-0-b64d5ed6.pth'
}

# Hardcoded model info
MODEL_INFO = {
    'yolov5s': {
        'description': '⚖️ YOLOv5 Small - Optimal untuk currency detection',
        'size': '14.4 MB',
        'category': 'detection',
        'filename': 'yolov5s.pt',
        'url_key': 'yolov5s'
    },
    'efficientnet': {
        'description': '🎯 EfficientNet - Lightweight model untuk klasifikasi',
        'size': '31 MB',
        'category': 'classification',
        'filename': 'efficientnet-b0.pth',
        'url_key': 'efficientnet'
    }
}

# Default directories
DEFAULT_DIRECTORIES = {
    'models': '/data/pretrained',
    'checkpoints': '/checkpoints',
    'downloads': '/downloads'
}


def get_default_config() -> Dict[str, Any]:
    """⚙️ Mendapatkan konfigurasi default
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    return DEFAULT_CONFIG.copy()


def get_default_directories() -> Dict[str, str]:
    """📁 Mendapatkan direktori default
    
    Returns:
        Dictionary berisi path direktori default
    """
    return DEFAULT_DIRECTORIES.copy()


def get_model_info() -> Dict[str, Any]:
    """ℹ️ Mendapatkan informasi model YOLOv5s
    
    Returns:
        Dictionary berisi informasi model
    """
    return {
        'type': 'yolov5s',
        'description': MODEL_INFO['yolov5s']['description'],
        'size': MODEL_INFO['yolov5s']['size'],
        'category': MODEL_INFO['yolov5s']['category'],
        'filename': MODEL_INFO['yolov5s']['filename']
    }


def _get_model_category() -> str:
    """🏷️ Mendapatkan kategori model
    
    Returns:
        String kategori model (always 'detection' for YOLOv5s)
    """
    return 'detection'