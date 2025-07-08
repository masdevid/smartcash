"""
File: smartcash/ui/dataset/preprocess/configs/preprocess_defaults.py
Description: Default configuration for preprocessing module
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocess.constants import (
    DEFAULT_SPLITS, VALIDATION_CONFIG, PERFORMANCE_CONFIG, YOLO_PRESETS
)

def get_default_preprocessing_config() -> Dict[str, Any]:
    """
    Get default preprocessing configuration.
    
    Returns:
        Default preprocessing configuration dictionary
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

# YAML configuration template
PREPROCESSING_CONFIG_YAML = """
# Preprocessing Configuration
preprocessing:
  target_splits: ['train', 'valid']
  
  validation:
    enabled: false              # Minimal validation only
    filename_pattern: true      # Auto-rename to research format
    auto_fix: true             # Auto-create directories
    
  normalization:
    preset: 'yolov5s'          # YOLO preset (yolov5s, yolov5m, yolov5l, yolov5x)
    target_size: [640, 640]    # Target image size
    preserve_aspect_ratio: true # Preserve aspect ratio with padding
    pixel_range: [0, 1]        # Pixel value range
    method: 'minmax'           # Normalization method
    
  batch_size: 32               # Processing batch size
  move_invalid: false          # Move invalid files to separate directory
  invalid_dir: 'data/invalid'  # Directory for invalid files
  cleanup_target: 'preprocessed' # Cleanup target (preprocessed, augmented, samples, both)
  backup_enabled: true         # Enable backup before processing

data:
  dir: 'data'                  # Source data directory
  preprocessed_dir: 'data/preprocessed' # Output directory

performance:
  batch_size: 32               # Processing batch size
  io_workers: 8                # Number of I/O workers
  cpu_workers: null            # Number of CPU workers (auto-detect)
  memory_limit_mb: 2048        # Memory limit in MB

ui:
  show_progress: true          # Show progress tracking
  show_details: true           # Show detailed information
  auto_scroll: true            # Auto-scroll logs
"""

# Form validation rules
FORM_VALIDATION_RULES = {
    'target_splits': {
        'required': True,
        'type': 'list',
        'min_length': 1,
        'allowed_values': ['train', 'valid', 'test']
    },
    'normalization.preset': {
        'required': True,
        'type': 'string',
        'allowed_values': list(YOLO_PRESETS.keys())
    },
    'normalization.target_size': {
        'required': True,
        'type': 'list',
        'length': 2,
        'item_type': 'int',
        'min_value': 32,
        'max_value': 2048
    },
    'batch_size': {
        'required': True,
        'type': 'int',
        'min_value': 1,
        'max_value': 256
    },
    'data.dir': {
        'required': True,
        'type': 'string',
        'min_length': 1
    }
}

# UI component default values
UI_DEFAULTS = {
    'resolution_dropdown': 'yolov5s',
    'normalization_dropdown': 'minmax',
    'preserve_aspect_checkbox': True,
    'target_splits_select': DEFAULT_SPLITS,
    'batch_size_input': 32,
    'validation_checkbox': False,
    'move_invalid_checkbox': False,
    'invalid_dir_input': 'data/invalid',
    'cleanup_target_dropdown': 'preprocessed',
    'backup_checkbox': True
}

# Configuration migration mappings
CONFIG_MIGRATIONS = {
    'v1.0.0': {
        'target_size': lambda x: [x, x] if isinstance(x, int) else x,
        'normalize': lambda x: {'enabled': x} if isinstance(x, bool) else x
    }
}

# Environment-specific defaults
ENVIRONMENT_DEFAULTS = {
    'colab': {
        'performance.io_workers': 4,
        'performance.memory_limit_mb': 1024,
        'ui.show_details': False
    },
    'local': {
        'performance.io_workers': 8,
        'performance.memory_limit_mb': 2048,
        'ui.show_details': True
    }
}