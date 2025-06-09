"""
File: smartcash/ui/dataset/augmentation/handlers/defaults.py
Deskripsi: Default config dengan intensity support sesuai augmentation_config.yaml
"""

from typing import Dict, Any
from datetime import datetime

def get_default_augmentation_config() -> Dict[str, Any]:
    """Default config dengan intensity support dan backend compatibility"""
    return {
        '_base_': 'base_config.yaml',
        'config_version': '2.1',  # Updated version dengan intensity
        'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        
        # Data configuration
        'data': {
            'dir': 'data',
            'splits': {
                'train': 'data/train',
                'valid': 'data/valid',
                'test': 'data/test'
            },
            'output': {
                'augmented': 'data/augmented',
                'preprocessed': 'data/preprocessed'
            }
        },
        
        # Augmentation config dengan intensity support
        'augmentation': {
            # Basic options (dari basic_opts_widget)
            'enabled': True,
            'num_variations': 2,          # IntSlider 1-10
            'target_count': 500,          # IntSlider 100-2000
            'intensity': 0.7,             # NEW: FloatSlider 0.1-1.0
            'output_prefix': 'aug',       # Text input
            'balance_classes': True,      # Checkbox
            'target_split': 'train',      # Dropdown
            'move_to_preprocessed': True,
            
            # Types (dari augtypes_opts_widget)
            'types': ['combined'],        # SelectMultiple
            'available_types': [
                'lighting', 'position', 'combined',
                'geometric', 'color', 'noise'
            ],
            
            # Position parameters (dari advanced_opts_widget)
            'position': {
                'horizontal_flip': 0.5,   # fliplr
                'rotation_limit': 12,     # degrees  
                'translate_limit': 0.08,  # translate
                'scale_limit': 0.04       # scale
            },
            
            # Lighting parameters
            'lighting': {
                'brightness_limit': 0.2,  # brightness
                'contrast_limit': 0.15,   # contrast
                'gamma_limit': [85, 115],
                'shadow_probability': 0.2
            },
            
            # Combined parameters (lighter versions)
            'combined': {
                'horizontal_flip': 0.5,
                'rotation_limit': 12,
                'translate_limit': 0.08,
                'scale_limit': 0.04,
                'brightness_limit': 0.2,
                'contrast_limit': 0.15,
                'gamma_limit': [85, 115],
                'shadow_probability': 0.2
            },
            
            # Processing settings
            'validate_outputs': True,
            'skip_existing': False,
            'resume': False,
            'output_dir': 'data/augmented'
        },
        
        # Preprocessing configuration
        'preprocessing': {
            'normalization': {
                'enabled': True,
                'method': 'minmax',       # dari norm_method dropdown
                'denormalize': False,     # dari denormalize checkbox
                'target_size': [640, 640],
                'preserve_aspect_ratio': False,
                'output_quality': 95,
                'save_float32': True,
                'save_visualization': True
            }
        },
        
        # Class balancing
        'balancing': {
            'enabled': True,
            'strategy': 'weighted',
            'layer_weights': {
                'layer1': 1.0,    # Banknote detection
                'layer2': 0.8,    # Nominal detection  
                'layer3': 0.5     # Security features
            },
            'max_files_per_class': 500,
            'min_files_per_class': 10
        },
        
        # File processing
        'file_processing': {
            'max_workers': 4,
            'batch_size': 100,
            'validate_images': True,
            'validate_labels': True,
            'check_bbox_validity': True,
            'skip_corrupted': True,
            'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
            'label_extension': '.txt'
        },
        
        # Progress tracking untuk dual progress tracker
        'progress': {
            'enabled': True,
            'granular_tracking': True,
            'operation_weights': {
                'validation': 10,
                'balancing': 15,
                'augmentation': 60,
                'normalization': 15
            },
            'update_frequency': 0.05,
            'log_frequency': 0.1
        },
        
        # Output configuration
        'output': {
            'create_split_dirs': True,
            'create_symlinks': True,
            'organize_by_split': True,
            'preserve_structure': True,
            'cleanup_on_error': True,
            'backup_original': False
        },
        
        # Logging
        'logging': {
            'level': 'INFO',
            'log_to_console': True,
            'log_progress': True,
            'log_statistics': True,
            'log_timing': True
        },
        
        # Validation dengan intensity range
        'validation': {
            'check_data_integrity': True,
            'validate_config': True,
            'verify_augmented_files': True,
            'check_symlink_integrity': True,
            'max_error_rate': 0.1,
            'min_success_rate': 0.8,
            
            # Form validation ranges dengan intensity
            'ranges': {
                'num_variations': [1, 10],
                'target_count': [100, 2000],
                'intensity': [0.1, 1.0],        # NEW: Intensity range
                'horizontal_flip': [0.0, 1.0],
                'rotation_limit': [0, 30],
                'translate_limit': [0.0, 0.25],
                'scale_limit': [0.0, 0.25],
                'brightness_limit': [0.0, 0.4],
                'contrast_limit': [0.0, 0.4]
            }
        },
        
        # Performance
        'performance': {
            'max_memory_usage_gb': 4.0,
            'enable_garbage_collection': True,
            'use_threading': True,
            'optimize_io': True,
            'cache_metadata': True,
            'cache_size_mb': 100
        },
        
        # Backend integration dengan intensity support
        'backend': {
            'service_enabled': True,
            'progress_tracking': True,
            'async_processing': False,
            'communicator_enabled': True,
            'intensity_scaling': True      # NEW: Intensity scaling support
        }
    }