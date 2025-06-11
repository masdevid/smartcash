"""
File: smartcash/ui/dataset/preprocessing/handlers/defaults.py
Deskripsi: Enhanced default configuration dengan multi-split dan validasi lengkap
"""

from typing import Dict, Any, List

def get_default_preprocessing_config() -> Dict[str, Any]:
    """Enhanced default configuration sesuai preprocessing_config.yaml structure"""
    return {
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'enabled': True,
            'output_dir': 'data/preprocessed',
            'save_visualizations': False,
            'vis_dir': 'visualizations/preprocessing',
            'sample_size': 0,
            'target_splits': ['train', 'valid'],  # Multi-select default
            'force_reprocess': False,
            
            'validation': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
                'invalid_dir': 'data/invalid',
                'visualize': False,
                'check_image_quality': True,
                'check_labels': True,
                'check_coordinates': True,
                'check_uuid_consistency': True
            },
            
            'normalization': {
                'enabled': True,
                'method': 'minmax',
                'target_size': [640, 640],
                'preserve_aspect_ratio': True,  # Enhanced dengan aspect ratio
                'normalize_pixel_values': True,
                'pixel_range': [0, 1]
            },
            
            'analysis': {
                'enabled': False,
                'class_balance': True,
                'image_size_distribution': True,
                'bbox_statistics': True,
                'layer_balance': True
            },
            
            'balance': {
                'enabled': False,
                'target_distribution': 'auto',
                'methods': {
                    'undersampling': False,
                    'oversampling': True,
                    'augmentation': True
                },
                'min_samples_per_class': 100,
                'max_samples_per_class': 1000
            }
        },
        
        'performance': {
            'batch_size': 32,  # Enhanced dengan batch size
            'use_gpu': True,
            'compression_level': 90,
            'max_memory_usage_gb': 4.0,
            'use_mixed_precision': True,
            
            'threading': {
                'io_workers': 8,
                'cpu_workers': None,
                'parallel_threshold': 100,
                'batch_processing': True
            }
        },
        
        'cleanup': {
            'augmentation_patterns': [
                'aug_.*',
                '.*_augmented.*',
                '.*_modified.*',
                '.*_processed.*',
                '.*_norm.*'
            ],
            'ignored_patterns': [
                '.*\.gitkeep',
                '.*\.DS_Store',
                '.*\.gitignore'
            ],
            'backup_dir': 'data/backup/preprocessing',
            'backup_enabled': False,
            'auto_cleanup_preprocessed': False
        }
    }

# Enhanced one-liner utilities
get_default_resolution = lambda: '640x640'
get_default_normalization = lambda: 'minmax'
get_default_batch_size = lambda: 32
get_default_splits = lambda: ['train', 'valid']
get_default_preserve_aspect = lambda: True
get_default_validation_enabled = lambda: True
get_default_move_invalid = lambda: True
get_default_invalid_dir = lambda: 'data/invalid'