"""
File: smartcash/ui/dataset/preprocessing/configs/defaults.py
Description: Provides the default configuration for the preprocessing module.
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocessing.constants import (
    DEFAULT_SPLITS, 
    VALIDATION_CONFIG, 
    PERFORMANCE_CONFIG
)

def get_default_config() -> Dict[str, Any]:
    """
    Returns the default configuration for the preprocessing module.

    This configuration includes settings for data paths, preprocessing parameters,
    performance, and UI behavior.

    Returns:
        A dictionary containing the default configuration.
    """
    return {
        'preprocessing': {
            'target_splits': DEFAULT_SPLITS.copy(),
            'validation': VALIDATION_CONFIG.copy(),
            'normalization': {
                'preset': 'yolov5s',
                'target_size': [640, 640],
                'preserve_aspect_ratio': True,
                'pixel_range': [0, 1],
                'method': 'minmax'
            },
            'batch_size': 32,
            'move_invalid': False,
            'invalid_dir': 'data/invalid',
            'cleanup_target': 'preprocessed',
            'backup_enabled': True
        },
        'data': {
            'dir': 'data',
            'preprocessed_dir': 'data/preprocessed'
        },
        'performance': PERFORMANCE_CONFIG.copy(),
        'ui': {
            'show_progress': True,
            'show_details': True,
            'auto_scroll': True
        }
    }

def get_default_preprocessing_config() -> Dict[str, Any]:
    """
    Alias for get_default_config() for compatibility.
    
    Returns:
        A dictionary containing the default preprocessing configuration.
    """
    return get_default_config()
