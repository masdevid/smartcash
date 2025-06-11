"""
File: smartcash/ui/dataset/preprocessing/handlers/defaults.py
Deskripsi: Default configuration sesuai API specifications dengan essential features only
"""

from typing import Dict, Any

def get_default_preprocessing_config() -> Dict[str, Any]:
    """Default config sesuai API requirements"""
    return {
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'enabled': True,
            'target_splits': ['train', 'valid'],
            'output_dir': 'data/preprocessed',
            'force_reprocess': False,
            
            'normalization': {
                'enabled': True,
                'method': 'minmax',
                'target_size': [640, 640],
                'preserve_aspect_ratio': True,
                'normalize_pixel_values': True,
                'pixel_range': [0, 1]
            },
            
            'validation': {
                'enabled': True,
                'move_invalid': True,
                'invalid_dir': 'data/invalid',
                'check_image_quality': True,
                'check_labels': True,
                'check_coordinates': True
            },
            
            'cleanup': {
                'target': 'preprocessed',
                'backup_enabled': False,
                'patterns': {
                    'preprocessed': ['pre_*.npy'],
                    'samples': ['sample_*.jpg']
                }
            }
        },
        
        'performance': {
            'batch_size': 32,
            'use_gpu': True,
            'max_memory_usage_gb': 4.0
        },
        
        'data': {
            'dir': 'data',
            'local': {
                'train': 'data/train',
                'valid': 'data/valid',
                'test': 'data/test'
            },
            'preprocessed_dir': 'data/preprocessed',
            'invalid_dir': 'data/invalid'
        },
        
        'file_naming': {
            'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
            'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'preserve_uuid': True
        },
        
        'api_settings': {
            'progress_tracking': True,
            'ui_integration': True,
            'enhanced_validation': True
        }
    }

# One-liner utilities
get_default_resolution = lambda: '640x640'
get_default_normalization = lambda: 'minmax'
get_default_batch_size = lambda: 32
get_default_splits = lambda: ['train', 'valid']
get_default_cleanup_target = lambda: 'preprocessed'