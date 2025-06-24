"""
File: smartcash/ui/dataset/preprocessing/handlers/config_updater.py
Deskripsi: UI updater untuk essential preprocessing forms
"""

from typing import Dict, Any

def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dengan config values"""
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    validation_config = preprocessing_config.get('validation', {})
    cleanup_config = preprocessing_config.get('cleanup', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update resolution
    try:
        target_size = normalization_config.get('target_size', [640, 640])
        resolution_str = f"{target_size[0]}x{target_size[1]}"
        valid_resolutions = ['320x320', '416x416', '512x512', '640x640', '832x832']
        safe_update('resolution_dropdown', resolution_str if resolution_str in valid_resolutions else '640x640')
    except Exception:
        safe_update('resolution_dropdown', '640x640')
    
    # Update normalization method
    try:
        normalization_enabled = normalization_config.get('enabled', True)
        method = normalization_config.get('method', 'minmax')
        final_value = 'none' if not normalization_enabled else method
        safe_update('normalization_dropdown', final_value if final_value in ['minmax', 'standard', 'none'] else 'minmax')
    except Exception:
        safe_update('normalization_dropdown', 'minmax')
    
    # Update target splits
    try:
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, list):
            validated_splits = [s for s in target_splits if s in ['train', 'valid', 'test']]
            safe_update('target_splits_select', tuple(validated_splits) if validated_splits else ('train', 'valid'))
        else:
            safe_update('target_splits_select', ('train', 'valid'))
    except Exception:
        safe_update('target_splits_select', ('train', 'valid'))
    
    # Update preserve aspect ratio
    try:
        preserve_aspect = normalization_config.get('preserve_aspect_ratio', True)
        safe_update('preserve_aspect_checkbox', bool(preserve_aspect))
    except Exception:
        safe_update('preserve_aspect_checkbox', True)
    
    # Update validation settings
    try:
        safe_update('validation_checkbox', bool(validation_config.get('enabled', True)))
        safe_update('move_invalid_checkbox', bool(validation_config.get('move_invalid', True)))
        safe_update('invalid_dir_input', str(validation_config.get('invalid_dir', 'data/invalid')))
    except Exception:
        safe_update('validation_checkbox', True)
        safe_update('move_invalid_checkbox', True)
        safe_update('invalid_dir_input', 'data/invalid')
    
    # Update cleanup settings
    try:
        cleanup_target = cleanup_config.get('target', 'preprocessed')
        valid_targets = ['preprocessed', 'samples', 'both']
        safe_update('cleanup_target_dropdown', cleanup_target if cleanup_target in valid_targets else 'preprocessed')
        safe_update('backup_checkbox', bool(cleanup_config.get('backup_enabled', False)))
    except Exception:
        safe_update('cleanup_target_dropdown', 'preprocessed')
        safe_update('backup_checkbox', False)
    
    # Update performance
    try:
        batch_size = performance_config.get('batch_size', 32)
        safe_update('batch_size_input', max(1, min(batch_size, 128)) if isinstance(batch_size, int) else 32)
    except Exception:
        safe_update('batch_size_input', 32)

def reset_preprocessing_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default values"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        default_config = get_default_preprocessing_config()
        update_preprocessing_ui(ui_components, default_config)
    except Exception:
        _apply_hardcoded_defaults(ui_components)

def _apply_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Hardcoded defaults fallback"""
    defaults = {
        'resolution_dropdown': '640x640',
        'normalization_dropdown': 'minmax',
        'target_splits_select': ('train', 'valid'),
        'preserve_aspect_checkbox': True,
        'validation_checkbox': True,
        'move_invalid_checkbox': True,
        'invalid_dir_input': 'data/invalid',
        'cleanup_target_dropdown': 'preprocessed',
        'backup_checkbox': False,
        'batch_size_input': 32
    }
    
    for key, value in defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass