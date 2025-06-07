
from typing import Dict, Any

def get_default_augmentation_config() -> Dict[str, Any]:
    """Default config sesuai dengan augmentation_config.yaml"""
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        
        # Konfigurasi augmentasi utama
        'augmentation': {
            # Parameter dasar
            'enabled': True,
            'types': ['combined', 'position', 'lighting'],
            'num_variations': 3,
            'target_count': 1000,
            'output_prefix': 'aug',
            'process_bboxes': True,
            'output_dir': 'data/augmented',
            'validate_results': True,
            'resume': False,
            'balance_classes': True,
            'move_to_preprocessed': True,
            
            # Parameter augmentasi posisi
            'position': {
                'fliplr': 0.5,
                'degrees': 15,
                'translate': 0.15,
                'scale': 0.15,
                'shear_max': 10
            },
            
            # Parameter augmentasi pencahayaan
            'lighting': {
                'hsv_h': 0.025,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'contrast': [0.7, 1.3],
                'brightness': [0.7, 1.3],
                'blur': 0.2,
                'noise': 0.1
            }
        },
        
        # Pengaturan pengelolaan data augmentasi
        'cleanup': {
            'backup_enabled': True,
            'backup_dir': 'data/backup/augmentation',
            'backup_count': 5,
            'patterns': ['aug_*', '*_augmented*']
        },
        
        # Pengaturan visualisasi
        'visualization': {
            'enabled': True,
            'sample_count': 5,
            'save_visualizations': True,
            'vis_dir': 'visualizations/augmentation',
            'show_original': True,
            'show_bboxes': True
        },
        
        # Pengaturan performa
        'performance': {
            'num_workers': 4,
            'batch_size': 16,
            'use_gpu': True
        }
    }