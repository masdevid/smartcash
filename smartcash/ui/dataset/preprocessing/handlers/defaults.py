
from typing import Dict, Any

def get_default_preprocessing_config() -> Dict[str, Any]:
    """
    Get hardcoded default configuration untuk downloader reset operations.
    Tidak bergantung pada yaml files untuk menghindari circular dependency.
    
    Returns:
        Dictionary berisi default configuration
    """
    return {
        'config_version': '1.0',
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'output_dir': 'data/preprocessed',
            'save_visualizations': True,
            'vis_dir': 'visualizations/preprocessing',
            'sample_size': 500,
            
            # Validasi dataset selama preprocessing
            'validate': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
                'visualize': True,
                'check_image_quality': True,
                'check_labels': True,
                'check_coordinates': True
            },
            
            # Opsi normalisasi
            'normalization': {
                'enabled': True,
                'method': 'minmax',
                'target_size': [640, 640],
                'preserve_aspect_ratio': True,
                'normalize_pixel_values': True,
                'pixel_range': [0, 1]
            },
            
            # Opsi analisis dataset
            'analysis': {
                'enabled': True,
                'class_balance': True,
                'image_size_distribution': True,
                'bbox_statistics': True,
                'layer_balance': True
            },
            
            # Opsi balancing dataset
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
        
        # Referensi ke konfigurasi augmentasi terpisah
        'augmentation_reference': {
            'config_file': 'augmentation_config.yaml',
            'use_for_preprocessing': True,
            'preprocessing_variations': 3
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
            'backup_dir': 'data/backup/preprocessing',
            'backup_enabled': False,
            'auto_cleanup_preprocessed': False
        },
        
        # Pengaturan performa preprocessing
        'performance': {
            'num_workers': 8,
            'batch_size': 32,
            'use_gpu': True,
            'compression_level': 90,
            'max_memory_usage_gb': 4.0,
            'use_mixed_precision': True
        }
    }