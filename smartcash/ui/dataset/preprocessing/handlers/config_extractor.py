"""
File: smartcash/ui/dataset/preprocessing/handlers/config_extractor.py
Deskripsi: Config extractor dengan API compatibility dan essential features only
"""

from typing import Dict, Any, List

def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config yang compatible dengan preprocessing API"""
    from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
    
    # Base dari defaults (DRY approach)
    config = get_default_preprocessing_config()
    
    # Helper untuk safe value extraction
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Extract core form values
    resolution = get_value('resolution_dropdown', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    # Extract target splits
    target_splits_widget = ui_components.get('target_splits_select')
    if target_splits_widget and hasattr(target_splits_widget, 'value'):
        target_splits = list(target_splits_widget.value) if target_splits_widget.value else ['train', 'valid']
    else:
        target_splits = ['train', 'valid']
    
    # Extract normalization settings
    normalization_method = get_value('normalization_dropdown', 'minmax')
    preserve_aspect = get_value('preserve_aspect_checkbox', True)
    
    # Extract validation settings  
    validation_enabled = get_value('validation_checkbox', True)
    move_invalid = get_value('move_invalid_checkbox', True)
    invalid_dir = get_value('invalid_dir_input', 'data/invalid').strip() or 'data/invalid'
    
    # Extract cleanup settings
    cleanup_target = get_value('cleanup_target_dropdown', 'preprocessed')
    backup_enabled = get_value('backup_checkbox', False)
    
    # Extract performance
    batch_size = max(1, min(get_value('batch_size_input', 32), 128))
    
    # Update config dengan form values
    config['preprocessing'].update({
        'target_splits': target_splits,
        'normalization': {
            'enabled': normalization_method != 'none',
            'method': normalization_method if normalization_method != 'none' else 'minmax',
            'target_size': [width, height],
            'preserve_aspect_ratio': preserve_aspect,
            'normalize_pixel_values': True,
            'pixel_range': [0, 1]
        },
        'validation': {
            'enabled': validation_enabled,
            'move_invalid': move_invalid,
            'invalid_dir': invalid_dir,
            'check_image_quality': True,
            'check_labels': True,
            'check_coordinates': True
        },
        'cleanup': {
            'target': cleanup_target,
            'backup_enabled': backup_enabled,
            'patterns': {
                'preprocessed': ['pre_*.npy'],
                'samples': ['sample_*.jpg']
            }
        }
    })
    
    config['performance'].update({
        'batch_size': batch_size,
        'use_gpu': True,
        'max_memory_usage_gb': 4.0
    })
    
    # Setup paths menggunakan environment manager
    config['data'] = _setup_environment_paths(target_splits)
    
    # API compatibility requirements
    config['file_naming'] = {
        'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
        'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'preserve_uuid': True
    }
    
    return config

def _setup_environment_paths(target_splits: List[str]) -> Dict[str, Any]:
    """Setup paths menggunakan environment manager"""
    try:
        from smartcash.common.environment import get_environment_manager
        
        env_manager = get_environment_manager()
        base_dir = str(env_manager.get_dataset_path())
        
        data_config = {
            'dir': base_dir,
            'local': {split: f"{base_dir}/{split}" for split in target_splits},
            'preprocessed_dir': f"{base_dir}/preprocessed",
            'invalid_dir': f"{base_dir}/invalid"
        }
        
        # Auto-create directories
        from pathlib import Path
        for split in target_splits:
            for subdir in ['images', 'labels']:
                Path(f"{base_dir}/{split}/{subdir}").mkdir(parents=True, exist_ok=True)
                Path(f"{base_dir}/preprocessed/{split}/{subdir}").mkdir(parents=True, exist_ok=True)
        
        Path(f"{base_dir}/invalid").mkdir(parents=True, exist_ok=True)
        
        return data_config
        
    except Exception:
        # Fallback ke basic setup
        base_dir = 'data'
        return {
            'dir': base_dir,
            'local': {split: f"{base_dir}/{split}" for split in target_splits},
            'preprocessed_dir': f"{base_dir}/preprocessed", 
            'invalid_dir': f"{base_dir}/invalid"
        }