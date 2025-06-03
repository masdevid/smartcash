"""
File: smartcash/ui/dataset/preprocessing/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi preprocessing dari UI components sesuai dengan preprocessing_config.yaml
"""

from typing import Dict, Any
from datetime import datetime


def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstraksi konfigurasi preprocessing yang konsisten dengan preprocessing_config.yaml"""
    # One-liner value extraction dengan fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Ekstrak nilai dari resolusi
    resolution = get_value('resolution_dropdown', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    # Metadata untuk config yang diperbarui
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Struktur konfigurasi sesuai dengan preprocessing_config.yaml
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'output_dir': get_value('output_dir', 'data/preprocessed'),
            'save_visualizations': get_value('save_visualizations', True),
            'vis_dir': get_value('vis_dir', 'visualizations/preprocessing'),
            'sample_size': get_value('sample_size', 500),
            
            # Validasi dataset selama preprocessing
            'validate': {
                'enabled': get_value('validate_enabled', True),
                'fix_issues': get_value('fix_issues', True),
                'move_invalid': get_value('move_invalid', True),
                'visualize': get_value('visualize', True),
                'check_image_quality': get_value('check_image_quality', True),
                'check_labels': get_value('check_labels', True),
                'check_coordinates': get_value('check_coordinates', True)
            },
            
            # Opsi normalisasi
            'normalization': {
                'enabled': get_value('normalization_enabled', True),
                'method': get_value('normalization_dropdown', 'minmax'),
                'target_size': [width, height],
                'preserve_aspect_ratio': get_value('preserve_aspect_ratio', True),
                'normalize_pixel_values': get_value('normalize_pixel_values', True),
                'pixel_range': [0, 1]
            },
            
            # Opsi analisis dataset
            'analysis': {
                'enabled': get_value('analysis_enabled', True),
                'class_balance': get_value('class_balance', True),
                'image_size_distribution': get_value('image_size_distribution', True),
                'bbox_statistics': get_value('bbox_statistics', True),
                'layer_balance': get_value('layer_balance', True)
            },
            
            # Opsi balancing dataset
            'balance': {
                'enabled': get_value('balance_enabled', False),
                'target_distribution': get_value('target_distribution', 'auto'),
                'methods': {
                    'undersampling': get_value('undersampling', False),
                    'oversampling': get_value('oversampling', True),
                    'augmentation': get_value('augmentation', True)
                },
                'min_samples_per_class': get_value('min_samples_per_class', 100),
                'max_samples_per_class': get_value('max_samples_per_class', 1000)
            }
        },
        
        # Referensi ke konfigurasi augmentasi terpisah
        'augmentation_reference': {
            'config_file': 'augmentation_config.yaml',
            'use_for_preprocessing': get_value('use_augmentation_for_preprocessing', True),
            'preprocessing_variations': get_value('preprocessing_variations', 3)
        },
        
        # Konfigurasi cleanup
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
            'backup_dir': get_value('backup_dir', 'data/backup/preprocessing'),
            'backup_enabled': get_value('backup_enabled', True),
            'auto_cleanup_preprocessed': get_value('auto_cleanup_preprocessed', False)
        },
        
        # Pengaturan performa preprocessing
        'performance': {
            'num_workers': get_value('worker_slider', 8),
            'batch_size': get_value('batch_size', 32),
            'use_gpu': get_value('use_gpu', True),
            'compression_level': get_value('compression_level', 90),
            'max_memory_usage_gb': get_value('max_memory_usage_gb', 4.0),
            'use_mixed_precision': get_value('use_mixed_precision', True)
        }
    }
