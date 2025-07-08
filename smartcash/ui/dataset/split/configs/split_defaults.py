"""
Default configurations and validation rules for dataset split module.

This module contains the default configuration and validation rules
for the dataset split functionality.
"""

# Default split configuration
DEFAULT_SPLIT_CONFIG = {
    'data': {
        'split_ratios': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        },
        'seed': 42,
        'shuffle': True,
        'stratify': False
    },
    'output': {
        'train_dir': 'data/train',
        'val_dir': 'data/val',
        'test_dir': 'data/test',
        'create_subdirs': True,
        'overwrite': False,
        'relative_paths': True,
        'preserve_dir_structure': True,
        'use_symlinks': False,
        'backup': True,
        'backup_dir': 'data/backup'
    }
}

# Validation rules for configuration
VALIDATION_RULES = {
    'required': [
        'data.input_dir',
        'data.output_dir',
        'data.split_ratios.train',
        'data.split_ratios.val',
        'data.split_ratios.test',
        'data.random_seed',
        'data.shuffle',
        'data.stratify',
        'data.overwrite',
        'data.backup',
        'data.backup_dir',
        'ui.show_advanced'
    ],
    'types': {
        'data.input_dir': str,
        'data.output_dir': str,
        'data.split_ratios.train': (int, float),
        'data.split_ratios.val': (int, float),
        'data.split_ratios.test': (int, float),
        'data.random_seed': int,
        'data.shuffle': bool,
        'data.stratify': bool,
        'data.overwrite': bool,
        'data.backup': bool,
        'data.backup_dir': str,
        'ui.show_advanced': bool
    },
    'constraints': {
        'split_ratios_sum': lambda ratios: 0.999 <= sum(ratios.values()) <= 1.001
    }
}
