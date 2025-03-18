"""
File: smartcash/ui/training_config/model_config_definitions.py
Deskripsi: Definisi default dan pemetaan konfigurasi model SmartCash
"""

from typing import Dict, Any

# Default konfigurasi model
DEFAULT_MODEL_CONFIG = {
    'model': {
        'model_type': 'efficient_optimized',
        'backbone': 'efficientnet_b4',
        'pretrained': True,
        'freeze_backbone': True,
        'use_attention': True,
        'use_residual': False,
        'use_ciou': False,
        'num_repeats': 3
    },
    'layers': {
        'banknote': {'enabled': True, 'threshold': 0.25},
        'nominal': {'enabled': True, 'threshold': 0.30},
        'security': {'enabled': True, 'threshold': 0.35}
    }
}

# Pemetaan model_type ke konfigurasi backbone dan fitur
MODEL_TYPE_CONFIGS = {
    'efficient_basic': {
        'backbone': 'efficientnet_b4',
        'use_attention': False,
        'use_residual': False,
        'use_ciou': False,
        'num_repeats': 1
    },
    'efficient_optimized': {
        'backbone': 'efficientnet_b4',
        'use_attention': True,
        'use_residual': False,
        'use_ciou': False,
        'num_repeats': 3
    },
    'efficient_advanced': {
        'backbone': 'efficientnet_b4',
        'use_attention': True,
        'use_residual': True,
        'use_ciou': True,
        'num_repeats': 3
    },
    'yolov5s': {
        'backbone': 'cspdarknet_s',
        'use_attention': False,
        'use_residual': False,
        'use_ciou': False,
        'num_repeats': 1
    },
    'efficient_experiment': {
        'backbone': 'efficientnet_b4',
        'use_attention': True,
        'use_residual': True,
        'use_ciou': True,
        'num_repeats': 5
    }
}

def get_default_config() -> Dict[str, Any]:
    """
    Dapatkan default konfigurasi model.
    
    Returns:
        Dictionary konfigurasi default
    """
    return DEFAULT_MODEL_CONFIG.copy()

def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi untuk model tertentu.
    
    Args:
        model_type: Tipe model (efficient_basic, efficient_optimized, dll)
    
    Returns:
        Dictionary konfigurasi model
    """
    return MODEL_TYPE_CONFIGS.get(model_type, MODEL_TYPE_CONFIGS['efficient_optimized']).copy()