"""
File: smartcash/ui/dataset/augmentation/handlers/defaults.py
Deskripsi: Default config minimal sesuai form fields yang ada
"""

from typing import Dict, Any

def get_default_augmentation_config() -> Dict[str, Any]:
    """Default config minimal fokus pada form fields dan backend essentials"""
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
            'output_prefix': 'aug',       # Text
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
            
            # Advanced form fields (lighting)
            'lighting': {
                'brightness_limit': 0.2,  # brightness slider
                'contrast_limit': 0.15    # contrast slider
            },
            
            # Combined params (sync dengan position + lighting)
            'combined': {
                'horizontal_flip': 0.5,
                'rotation_limit': 12,
                'translate_limit': 0.08,
                'scale_limit': 0.04,
                'brightness_limit': 0.2,
                'contrast_limit': 0.15
            }
        },
        
        # Normalization form fields
        'preprocessing': {
            'normalization': {
                'method': 'minmax',       # norm_method dropdown
                'denormalize': False,     # denormalize checkbox
                'target_size': [640, 640]
            }
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