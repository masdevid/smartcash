"""
File: smartcash/ui/dataset/preprocessing/handlers/config_updater.py
Deskripsi: Fixed config updater dengan proper inheritance handling
"""

from typing import Dict, Any

def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan inheritance handling"""
    # Extract sections dengan safe defaults
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update worker slider
    num_workers = performance_config.get('num_workers', 8)
    safe_update('worker_slider', min(max(num_workers, 1), 10))
    
    # Update split dropdown
    target_split = preprocessing_config.get('target_split', 'all')
    valid_splits = ['all', 'train', 'valid', 'test']
    safe_update('split_dropdown', target_split if target_split in valid_splits else 'all')
    
    # Update resolution dari target_size
    try:
        target_size = normalization_config.get('target_size', [640, 640])
        if isinstance(target_size, list) and len(target_size) >= 2:
            resolution_str = f"{target_size[0]}x{target_size[1]}"
            valid_resolutions = ['320x320', '416x416', '512x512', '640x640']
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

def reset_preprocessing_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke defaults"""
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
        'worker_slider': 8,
        'split_dropdown': 'all'
    }
    
    for key, value in defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass