"""
Constants for the Dataset Split module.

This module contains all the configuration constants used by the Dataset Split UI.
"""

# UI Configuration
UI_CONFIG = {
    'module_name': 'split',
    'parent_module': 'dataset',
    'title': 'Dataset Split',
    'subtitle': 'Configure how to split your dataset into train/validation/test sets',
    'description': 'Configure how to split your dataset into train/validation/test sets',
    'version': '1.0.0'
}

# Module Metadata
MODULE_METADATA = {
    'name': 'split',
    'title': '📊 Dataset Split',
    'description': 'Dataset splitting module with train/validation/test configuration',
    'version': '1.0.0',
    'category': 'dataset',
    'author': 'SmartCash',
    'tags': ['dataset', 'split', 'train', 'validation', 'test'],
    'features': [
        'Train/validation/test split configuration',
        'Custom split ratios',
        'Stratified and random splitting',
        'Directory structure preservation',
        'Backup and overwrite options',
        'Seed-based reproducible splits'
    ]
}

# Button Configuration
BUTTON_CONFIG = {
    'save': {'button_style': 'success', 'icon': 'save'},
    'reset': {'button_style': 'warning', 'icon': 'undo'},
    'split': {'button_style': 'success', 'icon': 'scissors'},
    'check': {'button_style': 'info', 'icon': 'check'},
    'cleanup': {'button_style': 'warning', 'icon': 'trash'}
}

# Default split ratios
DEFAULT_SPLIT_RATIOS = {
    'train': 0.7,
    'validation': 0.15,
    'test': 0.15
}

# Validation rules
VALIDATION_RULES = {
    'min_ratio': 0.0,
    'max_ratio': 1.0,
    'min_total': 0.95,  # 95% minimum total to account for rounding
    'max_total': 1.05   # 105% maximum total to account for rounding
}

# UI Constants
DEFAULT_LAYOUT = {
    'width': '100%',
    'padding': '10px',
    'margin': '5px 0',
    'border': '1px solid #e0e0e0',
    'border_radius': '5px'
}

# Module Metadata (backward compatibility)
MODULE_NAME = 'split'
MODULE_GROUP = 'dataset'
MODULE_TITLE = '📊 Dataset Split'
