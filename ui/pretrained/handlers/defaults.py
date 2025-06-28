# File: smartcash/ui/pretrained/handlers/defaults.py
"""
File: smartcash/ui/pretrained/handlers/defaults.py
Deskripsi: Default configuration dan constants untuk pretrained models
"""

from typing import Dict, Any, List

# Default configuration untuk pretrained models
DEFAULT_CONFIG = {
    'pretrained_models': {
        'models_dir': '/content/models',
        'drive_models_dir': '/data/pretrained',
        'pretrained_type': 'yolov5s',
        'auto_download': False,
        'sync_drive': True
    }
}

# Available model variants - Simplified to YOLOv5s only
MODEL_VARIANTS = ['yolov5s']

# Model descriptions - Single option
MODEL_DESCRIPTIONS = {
    'yolov5s': '⚖️ YOLOv5 Small - Optimal untuk currency detection'
}

# Model file sizes - Single option
MODEL_SIZES = {
    'yolov5s': '14.4 MB'
}

# Default directories
DEFAULT_DIRECTORIES = {
    'local_models': '/content/models',
    'drive_models': '/data/pretrained',
    'checkpoints': '/content/checkpoints',
    'downloads': '/content/downloads'
}


def get_default_pretrained_config() -> Dict[str, Any]:
    """🔧 Mendapatkan konfigurasi default untuk pretrained models
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    return DEFAULT_CONFIG.copy()


def get_model_variants() -> List[str]:
    """📋 Mendapatkan daftar variant model yang tersedia
    
    Returns:
        List berisi nama-nama variant model
    """
    return MODEL_VARIANTS.copy()


def get_model_descriptions() -> Dict[str, str]:
    """📝 Mendapatkan deskripsi untuk setiap variant model
    
    Returns:
        Dictionary berisi deskripsi model
    """
    return MODEL_DESCRIPTIONS.copy()


def get_model_sizes() -> Dict[str, str]:
    """📊 Mendapatkan ukuran file untuk setiap variant model
    
    Returns:
        Dictionary berisi ukuran file model
    """
    return MODEL_SIZES.copy()


def get_default_directories() -> Dict[str, str]:
    """📁 Mendapatkan direktori default
    
    Returns:
        Dictionary berisi path direktori default
    """
    return DEFAULT_DIRECTORIES.copy()


def get_model_info(model_type: str) -> Dict[str, str]:
    """ℹ️ Mendapatkan informasi lengkap untuk model tertentu
    
    Args:
        model_type: Tipe model (e.g., 'yolov5s')
        
    Returns:
        Dictionary berisi informasi model
    """
    if model_type not in MODEL_VARIANTS:
        model_type = 'yolov5s'  # Fallback ke default
    
    return {
        'type': model_type,
        'description': MODEL_DESCRIPTIONS.get(model_type, 'Model YOLOv5'),
        'size': MODEL_SIZES.get(model_type, 'Unknown'),
        'category': _get_model_category(model_type)
    }


def _get_model_category(model_type: str) -> str:
    """🏷️ Mendapatkan kategori model berdasarkan tipe
    
    Args:
        model_type: Tipe model
        
    Returns:
        String kategori model
    """
    # Simplified - hanya YOLOv5s yang tersedia
    return 'currency_detection_optimized' if model_type == 'yolov5s' else 'standard'