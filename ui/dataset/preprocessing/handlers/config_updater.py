"""
File: smartcash/ui/dataset/preprocessing/handlers/config_updater.py
Deskripsi: Pembaruan UI components dari konfigurasi preprocessing sesuai form yang ada
"""

from typing import Dict, Any

def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config preprocessing sesuai form yang tersedia"""
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Field mappings sesuai dengan form UI yang ada
    field_mappings = [
        # Resolution mapping dari target_size
        ('worker_slider', performance_config, 'num_workers', 8),
        ('split_dropdown', preprocessing_config, 'target_split', 'all'),
    ]
    
    # Apply mappings
    [safe_update(component_key, source_config.get(config_key, default_value)) 
     for component_key, source_config, config_key, default_value in field_mappings]
    
    # Special handling untuk resolution dropdown
    try:
        target_size = normalization_config.get('target_size', [640, 640])
        if isinstance(target_size, list) and len(target_size) >= 2:
            resolution_str = f"{target_size[0]}x{target_size[1]}"
            safe_update('resolution_dropdown', resolution_str)
    except Exception:
        safe_update('resolution_dropdown', '640x640')
    
    # Special handling untuk normalization method
    try:
        if normalization_config.get('enabled', True):
            method = normalization_config.get('method', 'minmax')
            safe_update('normalization_dropdown', method)
        else:
            safe_update('normalization_dropdown', 'none')
    except Exception:
        safe_update('normalization_dropdown', 'minmax')

def reset_preprocessing_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI components ke default konfigurasi preprocessing"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        default_config = get_default_preprocessing_config()
        update_preprocessing_ui(ui_components, default_config)
    except Exception:
        _apply_basic_defaults(ui_components)

def _apply_basic_defaults(ui_components: Dict[str, Any]) -> None:
    """Apply basic defaults jika config manager tidak tersedia"""
    basic_defaults = {
        'resolution_dropdown': '640x640',
        'normalization_dropdown': 'minmax',
        'worker_slider': 8,
        'split_dropdown': 'all'
    }
    
    for key, value in basic_defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass  # Silent fail untuk widget issues