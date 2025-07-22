"""
File: smartcash/ui/dataset/augmentation/configs/augmentation_defaults.py
Description: Default configurations for the augmentation module.
"""

from typing import Dict, Any, List

def get_default_augmentation_config() -> Dict[str, Any]:
    """
    Get the default configuration for the augmentation module.
    
    Returns:
        Dictionary containing default configuration values
    """
    return {
        # General settings
        'enabled': True,
        'augmentation_type': 'basic',  # basic, advanced, custom
        'intensity': 0.5,  # 0.0 to 1.0
        'random_seed': 42,
        
        # Input/Output settings
        'input_dir': '',
        'output_dir': '',
        'output_format': 'same_as_input',  # same_as_input, jpg, png, etc.
        'preserve_original': True,
        'create_subfolders': True,
        
        # Batch processing (standardized for better performance)
        'batch_size': 100,
        'num_workers': 4,
        'shuffle': True,
        
        # Augmentation pipeline
        'pipeline': [
            {
                'type': 'random_rotation',
                'enabled': True,
                'degrees': 15,
                'p': 0.5
            },
            {
                'type': 'random_flip',
                'enabled': True,
                'horizontal': True,
                'vertical': False,
                'p': 0.5
            },
            {
                'type': 'random_brightness',
                'enabled': True,
                'factor': 0.2,
                'p': 0.5
            },
            {
                'type': 'random_contrast',
                'enabled': True,
                'factor': 0.2,
                'p': 0.5
            }
        ],
        
        # Preview settings
        'preview_enabled': True,
        'preview_count': 5,
        'preview_grid_size': (3, 3),
        
        # Performance settings
        'use_gpu': True,
        'mixed_precision': True,
        'cache_augmented': False,
        
        # Logging and monitoring
        'log_level': 'INFO',
        'progress_bar': True,
        'save_augmentation_log': True,
        'log_dir': 'logs/augmentation',
        
        # Version and metadata
        'version': '1.0.0',
        'last_updated': '2023-01-01',
        'author': 'SmartCash Team'
    }

def get_default_config() -> Dict[str, Any]:
    """
    Alias for get_default_augmentation_config for backward compatibility.
    
    Returns:
        Dictionary containing default configuration values
    """
    return get_default_augmentation_config()

def get_augmentation_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined augmentation presets.
    
    Returns:
        Dictionary of preset configurations
    """
    return {
        'basic': {
            'description': 'Basic augmentations for general use',
            'pipeline': [
                {'type': 'random_rotation', 'degrees': 15, 'p': 0.3},
                {'type': 'random_flip', 'horizontal': True, 'p': 0.5}
            ]
        },
        'moderate': {
            'description': 'Moderate augmentations for varied datasets',
            'pipeline': [
                {'type': 'random_rotation', 'degrees': 30, 'p': 0.5},
                {'type': 'random_flip', 'horizontal': True, 'p': 0.5},
                {'type': 'random_brightness', 'factor': 0.2, 'p': 0.3},
                {'type': 'random_contrast', 'factor': 0.2, 'p': 0.3}
            ]
        },
        'aggressive': {
            'description': 'Aggressive augmentations for large datasets',
            'pipeline': [
                {'type': 'random_rotation', 'degrees': 45, 'p': 0.7},
                {'type': 'random_flip', 'horizontal': True, 'vertical': True, 'p': 0.5},
                {'type': 'random_brightness', 'factor': 0.3, 'p': 0.5},
                {'type': 'random_contrast', 'factor': 0.3, 'p': 0.5},
                {'type': 'random_saturation', 'factor': 0.3, 'p': 0.3},
                {'type': 'random_hue', 'factor': 0.1, 'p': 0.3}
            ]
        },
        'custom': {
            'description': 'Fully customizable augmentation pipeline',
            'pipeline': []
        }
    }
