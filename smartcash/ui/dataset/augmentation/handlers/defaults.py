"""
File: smartcash/ui/dataset/augmentation/handlers/defaults.py
Deskripsi: Default config untuk augmentation dengan mapping lengkap form UI
"""

from typing import Dict, Any
from datetime import datetime

def get_default_augmentation_config() -> Dict[str, Any]:
    """Default config dengan mapping lengkap ke UI form components"""
    return {
        '_base_': 'base_config.yaml',
        'config_version': '2.0',
        'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        
        # Augmentation config - mapping ke UI forms
        'augmentation': {
            # Basic options mapping (basic_opts_widget)
            'enabled': True,
            'num_variations': 3,          # IntSlider 1-10
            'target_count': 500,          # IntSlider 100-2000
            'output_prefix': 'aug',       # Text input
            'balance_classes': True,      # Checkbox
            'target_split': 'train',      # Dropdown
            'move_to_preprocessed': True,
            
            # Augmentation types mapping (augtypes_opts_widget)
            'types': ['combined'],        # SelectMultiple
            
            # Position parameters mapping (advanced_opts_widget position tab)
            'position': {
                'fliplr': 0.5,           # FloatSlider 0.0-1.0
                'degrees': 10,           # IntSlider 0-30
                'translate': 0.1,        # FloatSlider 0.0-0.25
                'scale': 0.1             # FloatSlider 0.0-0.25
            },
            
            # Lighting parameters mapping (advanced_opts_widget lighting tab)
            'lighting': {
                'hsv_h': 0.015,          # FloatSlider 0.0-0.05
                'hsv_s': 0.7,            # FloatSlider 0.0-1.0
                'brightness': 0.2,       # FloatSlider 0.0-0.4
                'contrast': 0.2          # FloatSlider 0.0-0.4
            },
            
            # Processing settings
            'process_bboxes': True,
            'validate_results': True,
            'resume': False,
            'output_dir': 'data/augmented',
            
            # Progress tracking untuk new progress tracker
            'pipeline_steps': ["prepare", "augment", "normalize", "verify"],
            'step_weights': {"prepare": 10, "augment": 50, "normalize": 30, "verify": 10}
        },
        
        # Cleanup config
        'cleanup': {
            'backup_enabled': True,
            'backup_dir': 'data/backup/augmentation',
            'backup_count': 5,
            'patterns': ['aug_*', '*_augmented*', '*_processed*'],
            'cleanup_steps': ["locate", "analyze", "backup", "execute"]
        },
        
        # Visualization config
        'visualization': {
            'enabled': True,
            'sample_count': 5,
            'save_visualizations': True,
            'vis_dir': 'visualizations/augmentation',
            'show_original': True,
            'show_bboxes': True
        },
        
        # Progress config untuk new progress tracker
        'progress': {
            'operations': {
                'augmentation': {
                    'steps': ["prepare", "augment", "normalize", "verify"],
                    'weights': {"prepare": 10, "augment": 50, "normalize": 30, "verify": 10},
                    'auto_hide': True
                },
                'check_dataset': {
                    'steps': ["locate", "analyze_raw", "analyze_augmented", "analyze_preprocessed"],
                    'weights': {"locate": 10, "analyze_raw": 30, "analyze_augmented": 30, "analyze_preprocessed": 30},
                    'auto_hide': False
                },
                'cleanup': {
                    'steps': ["locate", "analyze", "backup", "execute"],
                    'weights': {"locate": 10, "analyze": 20, "backup": 30, "execute": 40},
                    'auto_hide': True
                }
            },
            'display': {
                'level': 'triple',
                'show_step_info': False,
                'auto_hide_delay': 3600.0,
                'animation_speed': 0.1
            }
        },
        
        # Validation ranges untuk form validation
        'validation': {
            'ranges': {
                'num_variations': [1, 10],
                'target_count': [100, 2000],
                'fliplr': [0.0, 1.0],
                'degrees': [0, 30],
                'translate': [0.0, 0.25],
                'scale': [0.0, 0.25],
                'hsv_h': [0.0, 0.05],
                'hsv_s': [0.0, 1.0],
                'brightness': [0.0, 0.4],
                'contrast': [0.0, 0.4]
            },
            'required': ['num_variations', 'target_count', 'types', 'target_split'],
            'defaults': {
                'num_variations': 3,
                'target_count': 500,
                'output_prefix': 'aug',
                'balance_classes': True,
                'target_split': 'train',
                'types': ['combined']
            }
        }
    }