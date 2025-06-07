"""
File: smartcash/ui/dataset/preprocessing/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi preprocessing dari UI components sesuai form yang ada
"""

from typing import Dict, Any
from datetime import datetime

def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstraksi konfigurasi preprocessing sesuai dengan form UI yang ada"""
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract resolution value dan convert ke target_size
    resolution = get_value('resolution_dropdown', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'output_dir': 'data/preprocessed',
            'save_visualizations': False,
            'vis_dir': 'visualizations/preprocessing',
            'sample_size': 0,  # 0 = semua file
            
            # Validasi preprocessing
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
            
            # Normalisasi sesuai form UI
            'normalization': {
                'enabled': get_value('normalization_dropdown', 'minmax') != 'none',
                'method': get_value('normalization_dropdown', 'minmax'),
                'target_size': [width, height],
                'preserve_aspect_ratio': True,
                'normalize_pixel_values': True,
                'pixel_range': [0, 1]
            },
            
            # Split target dari form
            'target_split': get_value('split_dropdown', 'all'),
            'force_reprocess': False,
            
            # Analysis settings (default disabled)
            'analysis': {
                'enabled': False,
                'class_balance': True,
                'image_size_distribution': True,
                'bbox_statistics': True,
                'layer_balance': True
            },
            
            # Balance settings (default disabled)
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
        
        # Performance settings dari form
        'performance': {
            'num_workers': get_value('worker_slider', _get_optimal_workers()),
            'batch_size': 32,
            'use_gpu': True,
            'compression_level': 90,
            'max_memory_usage_gb': 4.0,
            'use_mixed_precision': True
        },
        
        # Cleanup settings
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

def _get_optimal_workers() -> int:
    """Get optimal workers untuk preprocessing operations"""
    from smartcash.common.threadpools import get_optimal_thread_count
    return get_optimal_thread_count('io')