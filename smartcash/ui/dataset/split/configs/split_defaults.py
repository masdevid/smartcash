"""
Default configurations and validation rules for dataset split module.

This module contains the default configuration and validation rules
for the dataset split functionality.
"""

from typing import Dict, Any

def get_default_split_config() -> Dict[str, Any]:
    """
    Get default split configuration.
    
    Returns:
        Default configuration dictionary for split module
    """
    return {
        'split': {
            'input_dir': 'data/raw',
            'output_dir': 'data/split',
            'ratios': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            },
            'method': 'random',  # 'random' or 'stratified'
            'seed': 42,
            'shuffle': True,
            'preserve_structure': True,
            'copy_files': True,  # vs move_files
            'create_dirs': True
        },
        'data': {
            'file_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
            'min_files_per_split': 1,
            'validate_images': True,
            'skip_corrupted': True
        },
        'output': {
            'train_dir': 'train',
            'val_dir': 'val', 
            'test_dir': 'test',
            'overwrite': False,
            'backup': True,
            'backup_dir': 'backup'
        },
        'advanced': {
            'use_relative_paths': True,
            'preserve_dir_structure': True,
            'create_symlinks': False,
            'parallel_processing': True,
            'batch_size': 1000,
            'progress_interval': 0.1
        },
        'ui': {
            'show_advanced': False,
            'auto_refresh': True,
            'preview_enabled': True
        }
    }

# Legacy support - keep old structure for backward compatibility
DEFAULT_SPLIT_CONFIG = get_default_split_config()

# Validation rules for configuration
VALIDATION_RULES = {
    'required': [
        'split.input_dir',
        'split.output_dir', 
        'split.ratios.train',
        'split.ratios.val',
        'split.ratios.test',
        'split.seed',
        'split.shuffle',
        'split.preserve_structure',
        'output.overwrite',
        'output.backup',
        'ui.show_advanced'
    ],
    'types': {
        'split.input_dir': str,
        'split.output_dir': str,
        'split.ratios.train': (int, float),
        'split.ratios.val': (int, float),
        'split.ratios.test': (int, float),
        'split.seed': int,
        'split.shuffle': bool,
        'split.preserve_structure': bool,
        'output.overwrite': bool,
        'output.backup': bool,
        'ui.show_advanced': bool
    },
    'constraints': {
        'split_ratios_sum': lambda config: 0.999 <= sum(config['split']['ratios'].values()) <= 1.001,
        'min_ratio_check': lambda config: all(ratio >= 0.0 for ratio in config['split']['ratios'].values()),
        'max_ratio_check': lambda config: all(ratio <= 1.0 for ratio in config['split']['ratios'].values())
    }
}

# Default button configurations
DEFAULT_BUTTON_CONFIG = {
    'split': {
        'label': 'Split Dataset',
        'description': 'Split dataset into train/validation/test sets',
        'button_style': 'primary',
        'icon': 'scissors',
        'tooltip': 'Execute dataset splitting operation'
    },
    'check': {
        'label': 'Check Status',
        'description': 'Check current split status and validation',
        'button_style': 'info', 
        'icon': 'check',
        'tooltip': 'Validate split configuration and check status'
    },
    'cleanup': {
        'label': 'Cleanup',
        'description': 'Clean up split directories',
        'button_style': 'warning',
        'icon': 'trash',
        'tooltip': 'Remove split directories and backup files'
    },
    'preview': {
        'label': 'Preview',
        'description': 'Preview split results',
        'button_style': 'success',
        'icon': 'eye',
        'tooltip': 'Preview how files will be split'
    }
}
