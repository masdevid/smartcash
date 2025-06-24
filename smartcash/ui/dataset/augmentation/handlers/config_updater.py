"""
File: smartcash/ui/dataset/augmentation/handlers/config_updater.py
Deskripsi: Config updater dengan HSV parameters dan cleanup target
"""

from typing import Dict, Any

def update_augmentation_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dengan HSV parameters dan cleanup target mapping"""
    # Extract sections dengan safe defaults
    aug_config = config.get('augmentation', {})
    position_config = aug_config.get('position', {})
    lighting_config = aug_config.get('lighting', {})
    cleanup_config = config.get('cleanup', {})
    
    # Safe update helper
    safe_update = lambda key, value: _safe_set_widget_value(ui_components, key, value)
    
    # Basic options mapping dengan cleanup target
    safe_update('num_variations', aug_config.get('num_variations', 2))
    safe_update('target_count', aug_config.get('target_count', 500))
    safe_update('intensity', aug_config.get('intensity', 0.7))
    safe_update('balance_classes', aug_config.get('balance_classes', True))
    safe_update('target_split', aug_config.get('target_split', 'train'))
    safe_update('cleanup_target', cleanup_config.get('default_target', 'both'))  # NEW
    
    # Augmentation types
    aug_types = aug_config.get('types', ['combined'])
    if isinstance(aug_types, list) and aug_types:
        safe_update('augmentation_types', aug_types)
    
    # Position parameters (mapping dari backend ke UI format)
    safe_update('fliplr', _validate_range(position_config.get('horizontal_flip', 0.5), 0.0, 1.0))
    safe_update('degrees', _validate_range(position_config.get('rotation_limit', 12), 0, 30))
    safe_update('translate', _validate_range(position_config.get('translate_limit', 0.08), 0.0, 0.25))
    safe_update('scale', _validate_range(position_config.get('scale_limit', 0.04), 0.0, 0.25))
    
    # Lighting parameters dengan HSV
    safe_update('brightness', _validate_range(lighting_config.get('brightness_limit', 0.2), 0.0, 0.4))
    safe_update('contrast', _validate_range(lighting_config.get('contrast_limit', 0.15), 0.0, 0.4))
    safe_update('hsv_h', _validate_range(lighting_config.get('hsv_hue', 10), 0, 30))
    safe_update('hsv_s', _validate_range(lighting_config.get('hsv_saturation', 15), 0, 50))
    
    # Backend integration notification
    _notify_backend_ui_update(ui_components, config)

def reset_augmentation_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI dengan HSV parameters dan cleanup target"""
    try:
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        default_config = get_default_augmentation_config()
        update_augmentation_ui(ui_components, default_config)
        
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "🔄 UI augmentasi direset ke default", "success")
        
        # Notify backend
        _notify_backend_ui_update(ui_components, default_config, reset=True)
            
    except Exception as e:
        _apply_hardcoded_defaults(ui_components)
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"⚠️ Error reset, fallback applied: {str(e)}", "warning")

def _safe_set_widget_value(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safe widget value setting dengan special handling"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            # SelectMultiple special handling
            if hasattr(widget, 'options') and isinstance(value, list):
                valid_options = [opt[1] if isinstance(opt, tuple) else opt for opt in widget.options]
                filtered_value = [v for v in value if v in valid_options]
                if filtered_value:
                    widget.value = filtered_value
            # Dropdown special handling
            elif hasattr(widget, 'options') and not isinstance(value, list):
                valid_options = [opt[1] if isinstance(opt, tuple) else opt for opt in widget.options]
                if value in valid_options:
                    widget.value = value
            else:
                widget.value = value
        except Exception:
            pass  # Silent fail untuk read-only atau incompatible widgets

def _validate_range(value: float, min_val: float, max_val: float) -> float:
    """Validate dan clamp value dalam range"""
    try:
        return max(min_val, min(max_val, float(value)))
    except (ValueError, TypeError):
        return min_val

def _notify_backend_ui_update(ui_components: Dict[str, Any], config: Dict[str, Any], reset: bool = False):
    """Notify backend tentang UI updates"""
    try:
        communicator = ui_components.get('backend_communicator')
        if communicator:
            action = "reset" if reset else "updated"
            communicator.log_info(f"🔄 UI {action} - backend config synchronized")
    except Exception:
        pass  # Silent fail

def _apply_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Hardcoded defaults fallback dengan HSV dan cleanup target"""
    defaults = {'num_variations': 2, 'target_count': 500, 'intensity': 0.7, 'balance_classes': True, 'target_split': 'train', 'cleanup_target': 'both', 'augmentation_types': ['combined'], 'fliplr': 0.5, 'degrees': 12, 'translate': 0.08, 'scale': 0.04, 'brightness': 0.2, 'contrast': 0.15, 'hsv_h': 10, 'hsv_s': 15}
    
    for key, value in defaults.items():
        _safe_set_widget_value(ui_components, key, value)