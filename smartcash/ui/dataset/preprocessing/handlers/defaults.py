"""
File: smartcash/ui/dataset/preprocessing/handlers/defaults.py
Deskripsi: Hardcoded default configuration untuk preprocessing reset operations
"""

from typing import Dict, Any

def get_default_preprocessing_config() -> Dict[str, Any]:
    """Get hardcoded default configuration untuk preprocessing reset operations"""
    return {
        'config_version': '1.0',
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'output_dir': 'data/preprocessed',
            'save_visualizations': False,
            'vis_dir': 'visualizations/preprocessing',
            'sample_size': 0,
            
            'validate': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
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
                'preserve_aspect_ratio': True,
                'normalize_pixel_values': True,
                'pixel_range': [0, 1]
            },
            
            'target_split': 'all',
            'force_reprocess': False,
            
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
            'num_workers': 8,
            'batch_size': 32,
            'use_gpu': True,
            'compression_level': 90,
            'max_memory_usage_gb': 4.0,
            'use_mixed_precision': True
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

# One-liner utilities
get_default_resolution = lambda: '640x640'
get_default_normalization = lambda: 'minmax'
get_default_workers = lambda: 8
get_default_split = lambda: 'all'
is_validation_enabled_by_default = lambda: True