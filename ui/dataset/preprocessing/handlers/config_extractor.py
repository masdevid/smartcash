"""
File: smartcash/ui/dataset/preprocessing/handlers/config_extractor.py
Deskripsi: Enhanced config extractor dengan API compatibility dan YOLO-specific features
"""

from typing import Dict, Any, List

def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced config extraction dengan API compatibility dan YOLO features"""
    from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
    
    # Base structure dari defaults (DRY)
    config = get_default_preprocessing_config()
    
    # Helper untuk get form values dengan safe fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Extract resolution dari dropdown
    resolution = get_value('resolution_dropdown', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    # Extract normalization method
    normalization_method = get_value('normalization_dropdown', 'minmax')
    
    # Extract multi-select target splits
    target_splits_widget = ui_components.get('target_splits_select')
    if target_splits_widget and hasattr(target_splits_widget, 'value'):
        target_splits = list(target_splits_widget.value) if target_splits_widget.value else ['train', 'valid']
    else:
        target_splits = ['train', 'valid']
    
    # Extract batch size
    batch_size = get_value('batch_size_input', 32)
    batch_size = max(1, min(batch_size, 128)) if isinstance(batch_size, int) else 32
    
    # Extract preserve aspect ratio checkbox
    preserve_aspect_ratio = get_value('preserve_aspect_checkbox', True)
    
    # Extract validation settings
    validation_enabled = get_value('validation_checkbox', True)
    move_invalid = get_value('move_invalid_checkbox', True)
    invalid_dir = get_value('invalid_dir_input', 'data/invalid').strip() or 'data/invalid'
    
    # 🔑 KEY: Use existing environment manager untuk path setup
    data_config = _setup_paths_with_environment_manager(target_splits)
    
    # Update form-controlled values dalam config
    config['preprocessing']['target_splits'] = target_splits
    config['preprocessing']['normalization']['enabled'] = normalization_method != 'none'
    config['preprocessing']['normalization']['method'] = normalization_method if normalization_method != 'none' else 'minmax'
    config['preprocessing']['normalization']['target_size'] = [width, height]
    config['preprocessing']['normalization']['preserve_aspect_ratio'] = preserve_aspect_ratio
    
    # 🎯 NEW: YOLO-specific normalization settings for API compatibility
    config['preprocessing']['normalization']['normalize_pixel_values'] = True
    config['preprocessing']['normalization']['pixel_range'] = [0, 1]
    
    # Update validation settings
    config['preprocessing']['validation']['enabled'] = validation_enabled
    config['preprocessing']['validation']['move_invalid'] = move_invalid
    config['preprocessing']['validation']['invalid_dir'] = invalid_dir
    
    # 🎯 NEW: Enhanced validation settings untuk API
    config['preprocessing']['validation']['check_image_quality'] = True
    config['preprocessing']['validation']['check_labels'] = True
    config['preprocessing']['validation']['check_coordinates'] = True
    config['preprocessing']['validation']['check_uuid_consistency'] = True
    
    # Update performance settings
    config['performance']['batch_size'] = batch_size
    
    # 🎯 NEW: Enhanced performance settings untuk API
    config['performance']['use_gpu'] = True
    config['performance']['compression_level'] = 90
    config['performance']['max_memory_usage_gb'] = 4.0
    config['performance']['use_mixed_precision'] = True
    
    # 🔑 KEY: Update data configuration dengan paths dari environment manager
    config['data'] = data_config
    
    # 🎯 CRITICAL: File naming configuration untuk API compatibility
    config['file_naming'] = {
        'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
        'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'augmented_pattern': 'aug_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'preserve_uuid': True,
        'auto_rename_to_raw': True
    }
    
    # 🎯 NEW: Output configuration untuk API
    config['preprocessing']['output'] = {
        'create_npy': True,
        'organize_by_split': True,
        'save_metadata': True
    }
    
    return config

def _setup_paths_with_environment_manager(target_splits: List[str]) -> Dict[str, Any]:
    """🔧 Setup paths menggunakan existing EnvironmentManager"""
    try:
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.constants.paths import get_paths_for_environment
        
        # Get environment manager
        env_manager = get_environment_manager()
        
        # Get system info
        system_info = env_manager.get_system_info()
        is_colab = env_manager.is_colab
        is_drive_mounted = env_manager.is_drive_mounted
        
        # Get appropriate paths
        paths = get_paths_for_environment(is_colab, is_drive_mounted)
        
        # Setup data config
        data_config = {
            'dir': str(env_manager.get_dataset_path()),
            'environment': system_info['environment'],
            'local': {},
            'paths_info': paths
        }
        
        # Setup paths untuk setiap split
        base_dir = data_config['dir']
        for split in target_splits:
            split_path = f"{base_dir}/{split}"
            data_config['local'][split] = split_path
            
            # Auto-create directories menggunakan pathlib (safer)
            try:
                from pathlib import Path
                Path(f"{split_path}/images").mkdir(parents=True, exist_ok=True)
                Path(f"{split_path}/labels").mkdir(parents=True, exist_ok=True)
            except Exception:
                pass  # Silent fail untuk permission issues
        
        # Setup output directories
        preprocessed_dir = f"{base_dir}/preprocessed"
        invalid_dir = f"{base_dir}/invalid"
        backup_dir = f"{base_dir}/backup/preprocessing"
        
        # Auto-create preprocessing directories
        try:
            from pathlib import Path
            for split in target_splits:
                Path(f"{preprocessed_dir}/{split}/images").mkdir(parents=True, exist_ok=True)
                Path(f"{preprocessed_dir}/{split}/labels").mkdir(parents=True, exist_ok=True)
            Path(invalid_dir).mkdir(parents=True, exist_ok=True)
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Silent fail untuk permission issues
        
        data_config.update({
            'preprocessed_dir': preprocessed_dir,
            'invalid_dir': invalid_dir,
            'backup_dir': backup_dir,
            'drive_mounted': is_drive_mounted,
            'colab_environment': is_colab
        })
        
        return data_config
        
    except Exception as e:
        # Fallback ke basic setup jika environment manager gagal
        return _fallback_path_setup(target_splits, str(e))

def _fallback_path_setup(target_splits: List[str], error_reason: str) -> Dict[str, Any]:
    """Fallback path setup jika environment manager tidak tersedia"""
    import os
    
    # Simple detection fallback
    is_colab = os.path.exists('/content')
    base_dir = '/content/data' if is_colab else 'data'
    
    data_config = {
        'dir': base_dir,
        'environment': 'colab_fallback' if is_colab else 'local_fallback',
        'local': {},
        'fallback_reason': error_reason
    }
    
    # Setup basic paths
    for split in target_splits:
        data_config['local'][split] = f"{base_dir}/{split}"
    
    data_config.update({
        'preprocessed_dir': f"{base_dir}/preprocessed",
        'invalid_dir': f"{base_dir}/invalid",
        'backup_dir': f"{base_dir}/backup/preprocessing"
    })
    
    return data_config

def get_environment_info() -> Dict[str, Any]:
    """Get environment info menggunakan existing environment manager"""
    try:
        from smartcash.common.environment import get_environment_manager
        
        env_manager = get_environment_manager()
        system_info = env_manager.get_system_info()
        
        return {
            'is_colab': env_manager.is_colab,
            'is_drive_mounted': env_manager.is_drive_mounted,
            'environment_type': system_info['environment'],
            'base_directory': system_info['base_directory'],
            'data_directory': system_info['data_directory'],
            'drive_path': system_info.get('drive_path'),
            'python_version': system_info.get('python_version'),
            'cuda_available': system_info.get('cuda_available', False)
        }
        
    except Exception as e:
        # Fallback info
        import os
        return {
            'is_colab': os.path.exists('/content'),
            'is_drive_mounted': False,
            'environment_type': 'unknown',
            'error': str(e),
            'fallback': True
        }

def validate_paths_exist(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate jika paths dalam config benar-benar ada dan create jika missing"""
    validation_result = {
        'valid': True,
        'missing_paths': [],
        'created_paths': [],
        'errors': []
    }
    
    try:
        from pathlib import Path
        
        data_config = config.get('data', {})
        local_paths = data_config.get('local', {})
        
        for split, path in local_paths.items():
            img_dir = Path(path) / 'images'
            label_dir = Path(path) / 'labels'
            
            for dir_path in [img_dir, label_dir]:
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        validation_result['created_paths'].append(str(dir_path))
                    except Exception as e:
                        validation_result['missing_paths'].append(str(dir_path))
                        validation_result['errors'].append(f"Cannot create {dir_path}: {str(e)}")
                        validation_result['valid'] = False
        
        # Check preprocessed directory
        preprocessing_config = config.get('preprocessing', {})
        output_dir = Path(preprocessing_config.get('output_dir', 'data/preprocessed'))
        
        if not output_dir.exists():
            try:
                for split in data_config.get('local', {}).keys():
                    split_dirs = [output_dir / split / 'images', output_dir / split / 'labels']
                    for dir_path in split_dirs:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        validation_result['created_paths'].append(str(dir_path))
            except Exception as e:
                validation_result['errors'].append(f"Cannot create output dir {output_dir}: {str(e)}")
                validation_result['valid'] = False
        
        # Check backup directory
        backup_dir = Path(data_config.get('backup_dir', 'data/backup/preprocessing'))
        if not backup_dir.exists():
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                validation_result['created_paths'].append(str(backup_dir))
            except Exception as e:
                validation_result['errors'].append(f"Cannot create backup dir {backup_dir}: {str(e)}")
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Path validation error: {str(e)}")
    
    return validation_result

def validate_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """🔍 Validate preprocessing config untuk API compatibility"""
    try:
        from smartcash.dataset.preprocessor.utils.config_validator import validate_preprocessing_config as api_validator
        return api_validator(config)
    except ImportError:
        # Fallback validation
        return _basic_config_validation(config)

def _basic_config_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Basic config validation sebagai fallback"""
    validation_result = {'valid': True, 'errors': [], 'warnings': []}
    
    try:
        # Validate preprocessing section
        preprocessing = config.get('preprocessing', {})
        if not preprocessing.get('enabled', True):
            validation_result['warnings'].append("Preprocessing disabled")
        
        # Validate target_size
        normalization = preprocessing.get('normalization', {})
        target_size = normalization.get('target_size', [640, 640])
        if not isinstance(target_size, list) or len(target_size) != 2:
            validation_result['errors'].append("Invalid target_size format")
            validation_result['valid'] = False
        
        # Validate target_splits
        target_splits = preprocessing.get('target_splits', [])
        if not target_splits:
            validation_result['errors'].append("No target splits specified")
            validation_result['valid'] = False
        
        # Validate data paths
        data_config = config.get('data', {})
        if not data_config.get('dir'):
            validation_result['errors'].append("No data directory specified")
            validation_result['valid'] = False
        
        # Validate file_naming
        file_naming = config.get('file_naming', {})
        required_patterns = ['raw_pattern', 'preprocessed_pattern']
        for pattern in required_patterns:
            if not file_naming.get(pattern):
                validation_result['warnings'].append(f"Missing {pattern} in file_naming")
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Validation error: {str(e)}")
    
    return validation_result

def get_preprocessing_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """📊 Get preprocessing config summary untuk UI display"""
    try:
        preprocessing = config.get('preprocessing', {})
        normalization = preprocessing.get('normalization', {})
        validation = preprocessing.get('validation', {})
        performance = config.get('performance', {})
        
        summary = {
            'enabled': preprocessing.get('enabled', True),
            'target_splits': preprocessing.get('target_splits', []),
            'resolution': f"{normalization.get('target_size', [640, 640])[0]}x{normalization.get('target_size', [640, 640])[1]}",
            'normalization_method': normalization.get('method', 'minmax'),
            'preserve_aspect_ratio': normalization.get('preserve_aspect_ratio', True),
            'validation_enabled': validation.get('enabled', True),
            'batch_size': performance.get('batch_size', 32),
            'use_gpu': performance.get('use_gpu', True),
            'total_splits': len(preprocessing.get('target_splits', [])),
            'api_compatible': True
        }
        
        return summary
        
    except Exception as e:
        return {
            'enabled': False,
            'error': str(e),
            'api_compatible': False
        }

# One-liner utilities untuk config manipulation
extract_target_splits = lambda config: config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
extract_resolution = lambda config: config.get('preprocessing', {}).get('normalization', {}).get('target_size', [640, 640])
extract_batch_size = lambda config: config.get('performance', {}).get('batch_size', 32)
extract_data_dir = lambda config: config.get('data', {}).get('dir', 'data')
is_validation_enabled = lambda config: config.get('preprocessing', {}).get('validation', {}).get('enabled', True)
is_normalization_enabled = lambda config: config.get('preprocessing', {}).get('normalization', {}).get('enabled', True)