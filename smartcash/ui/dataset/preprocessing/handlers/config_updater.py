"""
File: smartcash/ui/dataset/preprocessing/handlers/config_updater.py
Deskripsi: Enhanced UI updater dengan multi-split, validasi, dan aspect ratio support
"""

from typing import Dict, Any, List

def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Enhanced UI update dengan inheritance handling dan form validation"""
    # Extract sections dengan safe defaults (inheritance handling)
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    validation_config = preprocessing_config.get('validation', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update resolution dari target_size
    try:
        target_size = normalization_config.get('target_size', [640, 640])
        if isinstance(target_size, list) and len(target_size) >= 2:
            resolution_str = f"{target_size[0]}x{target_size[1]}"
            valid_resolutions = ['320x320', '416x416', '512x512', '640x640', '832x832']
            safe_update('resolution_dropdown', resolution_str if resolution_str in valid_resolutions else '640x640')
        else:
            safe_update('resolution_dropdown', '640x640')
    except Exception:
        safe_update('resolution_dropdown', '640x640')
    
    # Update normalization method
    try:
        normalization_enabled = normalization_config.get('enabled', True)
        normalization_method = normalization_config.get('method', 'minmax')
        
        if not normalization_enabled:
            safe_update('normalization_dropdown', 'none')
        else:
            valid_methods = ['minmax', 'standard', 'none']
            safe_update('normalization_dropdown', normalization_method if normalization_method in valid_methods else 'minmax')
    except Exception:
        safe_update('normalization_dropdown', 'minmax')
    
    # Update multi-select target splits
    try:
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, list):
            # Validate splits
            valid_splits = ['train', 'valid', 'test']
            validated_splits = [s for s in target_splits if s in valid_splits]
            validated_splits = validated_splits if validated_splits else ['train', 'valid']
            safe_update('target_splits_select', tuple(validated_splits))
        else:
            safe_update('target_splits_select', ('train', 'valid'))
    except Exception:
        safe_update('target_splits_select', ('train', 'valid'))
    
    # Update batch size
    try:
        batch_size = performance_config.get('batch_size', 32)
        batch_size = max(1, min(batch_size, 128)) if isinstance(batch_size, int) else 32
        safe_update('batch_size_input', batch_size)
    except Exception:
        safe_update('batch_size_input', 32)
    
    # Update preserve aspect ratio checkbox
    try:
        preserve_aspect_ratio = normalization_config.get('preserve_aspect_ratio', True)
        safe_update('preserve_aspect_checkbox', bool(preserve_aspect_ratio))
    except Exception:
        safe_update('preserve_aspect_checkbox', True)
    
    # Update validation settings
    try:
        validation_enabled = validation_config.get('enabled', True)
        safe_update('validation_checkbox', bool(validation_enabled))
        
        move_invalid = validation_config.get('move_invalid', True)
        safe_update('move_invalid_checkbox', bool(move_invalid))
        
        invalid_dir = validation_config.get('invalid_dir', 'data/invalid')
        safe_update('invalid_dir_input', str(invalid_dir))
    except Exception:
        safe_update('validation_checkbox', True)
        safe_update('move_invalid_checkbox', True)
        safe_update('invalid_dir_input', 'data/invalid')

def reset_preprocessing_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke enhanced defaults"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        default_config = get_default_preprocessing_config()
        update_preprocessing_ui(ui_components, default_config)
    except Exception:
        _apply_enhanced_hardcoded_defaults(ui_components)

def _apply_enhanced_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Enhanced hardcoded defaults fallback"""
    defaults = {
        'resolution_dropdown': '640x640',
        'normalization_dropdown': 'minmax',
        'target_splits_select': ('train', 'valid'),
        'batch_size_input': 32,
        'preserve_aspect_checkbox': True,
        'validation_checkbox': True,
        'move_invalid_checkbox': True,
        'invalid_dir_input': 'data/invalid'
    }
    
    for key, value in defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass