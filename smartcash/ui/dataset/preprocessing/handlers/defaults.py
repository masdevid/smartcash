"""
File: smartcash/ui/dataset/preprocessing/handlers/defaults.py
Deskripsi: Enhanced default configuration dengan API compatibility dan YOLO-specific features
"""

from typing import Dict, Any, List

def get_default_preprocessing_config() -> Dict[str, Any]:
    """🎯 Enhanced default configuration sesuai API specifications dan YOLO requirements"""
    return {
        '_base_': 'base_config.yaml',
        
        'preprocessing': {
            'enabled': True,
            'output_dir': 'data/preprocessed',
            'save_visualizations': False,
            'vis_dir': 'visualizations/preprocessing',
            'sample_size': 0,
            'target_splits': ['train', 'valid'],  # Multi-select default
            'force_reprocess': False,
            
            'validation': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
                'invalid_dir': 'data/invalid',
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
                'preserve_aspect_ratio': True,  # YOLO-specific
                'normalize_pixel_values': True,  # API requirement
                'pixel_range': [0, 1]  # API requirement
            },
            
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
            },
            
            # 🎯 NEW: Output configuration untuk API
            'output': {
                'create_npy': True,
                'organize_by_split': True,
                'save_metadata': True
            }
        },
        
        'performance': {
            'batch_size': 32,  # Enhanced dengan batch size
            'use_gpu': True,
            'compression_level': 90,
            'max_memory_usage_gb': 4.0,
            'use_mixed_precision': True,
            
            'threading': {
                'io_workers': 8,
                'cpu_workers': None,
                'parallel_threshold': 100,
                'batch_processing': True
            }
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
        },
        
        # 🎯 CRITICAL: File naming configuration untuk API compatibility
        'file_naming': {
            'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
            'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'augmented_pattern': 'aug_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'preserve_uuid': True,
            'auto_rename_to_raw': True
        },
        
        # 🎯 NEW: Data configuration template
        'data': {
            'dir': 'data',
            'local': {
                'train': 'data/train',
                'valid': 'data/valid',
                'test': 'data/test'
            },
            'preprocessed_dir': 'data/preprocessed',
            'invalid_dir': 'data/invalid',
            'backup_dir': 'data/backup/preprocessing'
        }
    }

def get_yolo_specific_config() -> Dict[str, Any]:
    """🎯 YOLO-specific configuration enhancements"""
    return {
        'preprocessing': {
            'normalization': {
                'method': 'minmax',
                'target_size': [640, 640],  # YOLO standard
                'preserve_aspect_ratio': True,
                'normalize_pixel_values': True,
                'pixel_range': [0, 1],
                'padding_color': [114, 114, 114]  # Gray padding
            },
            'validation': {
                'check_bbox_format': 'yolo',  # YOLO format validation
                'check_class_ids': True,
                'validate_coordinates': True,
                'check_file_pairs': True
            },
            'output': {
                'format': 'yolo',
                'create_npy': True,  # Normalized arrays
                'save_original_metadata': True,
                'coordinate_format': 'normalized'
            }
        }
    }

def get_api_compatibility_config() -> Dict[str, Any]:
    """🔗 API compatibility configuration"""
    return {
        'api_settings': {
            'consolidated_api': True,
            'progress_tracking': True,
            'ui_integration': True,
            'enhanced_validation': True,
            'yolo_optimization': True
        },
        'features': {
            'real_time_progress': True,
            'milestone_logging': True,
            'error_recovery': True,
            'batch_processing': True,
            'memory_optimization': True
        }
    }

def get_environment_specific_config(is_colab: bool = False, is_drive_mounted: bool = False) -> Dict[str, Any]:
    """🌍 Environment-specific configuration"""
    if is_colab and is_drive_mounted:
        return {
            'data': {
                'dir': '/content/drive/MyDrive/SmartCash/data',
                'environment': 'colab_drive'
            },
            'performance': {
                'use_gpu': True,
                'max_memory_usage_gb': 12.0,  # Colab Pro
                'batch_size': 64
            }
        }
    elif is_colab:
        return {
            'data': {
                'dir': '/content/data',
                'environment': 'colab_local'
            },
            'performance': {
                'use_gpu': True,
                'max_memory_usage_gb': 8.0,  # Colab free
                'batch_size': 32
            }
        }
    else:
        return {
            'data': {
                'dir': 'data',
                'environment': 'local'
            },
            'performance': {
                'use_gpu': False,
                'max_memory_usage_gb': 4.0,
                'batch_size': 16
            }
        }

def merge_config_enhancements(base_config: Dict[str, Any], *enhancements) -> Dict[str, Any]:
    """🔄 Merge configuration enhancements dengan base config"""
    import copy
    
    merged = copy.deepcopy(base_config)
    
    for enhancement in enhancements:
        merged = _deep_merge_dict(merged, enhancement)
    
    return merged

def _deep_merge_dict(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries"""
    import copy
    result = copy.deepcopy(base)
    
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result

def get_comprehensive_default_config(is_colab: bool = False, is_drive_mounted: bool = False) -> Dict[str, Any]:
    """🎯 Get comprehensive default config dengan semua enhancements"""
    base_config = get_default_preprocessing_config()
    yolo_config = get_yolo_specific_config()
    api_config = get_api_compatibility_config()
    env_config = get_environment_specific_config(is_colab, is_drive_mounted)
    
    return merge_config_enhancements(base_config, yolo_config, api_config, env_config)

# Enhanced one-liner utilities
get_default_resolution = lambda: '640x640'
get_default_normalization = lambda: 'minmax'
get_default_batch_size = lambda: 32
get_default_splits = lambda: ['train', 'valid']
get_default_preserve_aspect = lambda: True
get_default_validation_enabled = lambda: True
get_default_move_invalid = lambda: True
get_default_invalid_dir = lambda: 'data/invalid'
get_yolo_target_size = lambda: [640, 640]
get_yolo_pixel_range = lambda: [0, 1]
get_api_file_patterns = lambda: {
    'raw': 'rp_{nominal}_{uuid}_{sequence}',
    'preprocessed': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
    'augmented': 'aug_rp_{nominal}_{uuid}_{sequence}_{variance}'
}

# Configuration validation
def validate_default_config() -> Dict[str, Any]:
    """🔍 Validate default configuration"""
    try:
        config = get_default_preprocessing_config()
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required sections
        required_sections = ['preprocessing', 'performance', 'file_naming', 'data']
        for section in required_sections:
            if section not in config:
                validation_result['errors'].append(f"Missing required section: {section}")
                validation_result['valid'] = False
        
        # Check preprocessing subsections
        if 'preprocessing' in config:
            preprocessing = config['preprocessing']
            required_subsections = ['normalization', 'validation']
            for subsection in required_subsections:
                if subsection not in preprocessing:
                    validation_result['warnings'].append(f"Missing preprocessing subsection: {subsection}")
        
        # Validate target_size
        if 'preprocessing' in config and 'normalization' in config['preprocessing']:
            target_size = config['preprocessing']['normalization'].get('target_size')
            if not isinstance(target_size, list) or len(target_size) != 2:
                validation_result['errors'].append("Invalid target_size format")
                validation_result['valid'] = False
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': []
        }

# Export functions untuk backward compatibility
get_preprocessing_defaults = get_default_preprocessing_config
get_yolo_defaults = get_yolo_specific_config
get_api_defaults = get_api_compatibility_config