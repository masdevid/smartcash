"""
File: smartcash/ui/dataset/preprocessing/configs/defaults.py
Deskripsi: Default configuration untuk preprocessing module.
"""

from typing import Dict, Any


def get_default_preprocessing_config() -> Dict[str, Any]:
    """Get default preprocessing configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'preprocessing': {
            'enabled': True,
            'resolution': '640x640',
            'normalization': 'minmax',
            'preserve_aspect': True,
            'target_splits': ['train', 'valid'],
            'batch_size': 32,
            'validation': True,
            'move_invalid': True,
            'invalid_dir': 'invalid'
        },
        'cleanup': {
            'target': 'preprocessed',
            'backup': True
        },
        'data': {
            'dir': 'data'
        }
    }
