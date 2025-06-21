"""
File: smartcash/ui/dataset/augmentation/handlers/defaults.py
Deskripsi: Updated default config dengan HSV parameters dan cleanup target
"""

from typing import Dict, Any

def get_default_augmentation_config() -> Dict[str, Any]:
    """Default config dengan HSV parameters dan cleanup target untuk service compatibility"""
    return {
        # Data paths - backend essentials
        'data': {
            'dir': 'data'
        },
        
        # Form fields mapping
        'augmentation': {
            # Basic form fields
            'num_variations': 2,          # IntSlider
            'target_count': 500,          # IntSlider  
            'intensity': 0.7,             # FloatSlider
            'balance_classes': True,      # Checkbox
            'target_split': 'train',      # Dropdown
            'types': ['combined'],        # SelectMultiple
            
            # Advanced form fields (position)
            'position': {
                'horizontal_flip': 0.5,   # fliplr slider
                'rotation_limit': 12,     # degrees slider
                'translate_limit': 0.08,  # translate slider
                'scale_limit': 0.04       # scale slider
            },
            
            # Advanced form fields (lighting) - UPDATED: Added HSV parameters
            'lighting': {
                'brightness_limit': 0.2,  # brightness slider
                'contrast_limit': 0.15,   # contrast slider
                'hsv_hue': 10,            # hsv_h slider  
                'hsv_saturation': 15      # hsv_s slider
            },
            
            # Combined params (sync dengan position + lighting) - UPDATED: Added HSV
            'combined': {
                'horizontal_flip': 0.5,
                'rotation_limit': 12,
                'translate_limit': 0.08,
                'scale_limit': 0.04,
                'brightness_limit': 0.2,
                'contrast_limit': 0.15,
                'hsv_hue': 10,
                'hsv_saturation': 15
            }
        },
        
        # UPDATED: Cleanup configuration menggantikan preprocessing.normalization
        'cleanup': {
            'default_target': 'both',     # cleanup_target dropdown
            'confirm_before_cleanup': True,
            'backup_before_cleanup': False,
            'cleanup_empty_dirs': True,
            
            # Target-specific settings
            'targets': {
                'augmented': {
                    'include_preprocessed': True,
                    'patterns': ['aug_*']
                },
                'samples': {
                    'patterns': ['sample_aug_*'],
                    'preserve_originals': True
                },
                'both': {
                    'sequential': True
                }
            }
        },
        
        # Backend structure yang diharapkan service
        'backend': {
            'service_enabled': True,
            'progress_tracking': True,
            'async_processing': False,
            'max_workers': 4,
            'timeout_seconds': 300,
            'retry_count': 3,
            'validation_enabled': True
        },
        
        # Backend essentials only
        'balancing': {
            'enabled': True,
            'layer_weights': {'layer1': 1.0, 'layer2': 0.8, 'layer3': 0.5}
        },
        
        'file_processing': {
            'max_workers': 4,
            'batch_size': 100
        },
        
        'performance': {
            'num_workers': 4
        }
    }