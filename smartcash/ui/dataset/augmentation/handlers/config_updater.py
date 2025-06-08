"""
File: smartcash/ui/dataset/augmentation/handlers/config_updater.py
Deskripsi: Config updater dengan inheritance handling
"""

from typing import Dict, Any

def update_augmentation_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dengan inheritance handling"""
    # TODO: Implement UI update
    pass

def reset_augmentation_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke defaults"""
    # TODO: Implement UI reset
    pass
"""
File: smartcash/ui/dataset/augmentation/handlers/config_updater.py
Deskripsi: Config updater dengan inheritance handling dan safe form updates
"""

from typing import Dict, Any

def update_augmentation_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    CRITICAL: Update UI dengan inheritance handling dan safe form access
    
    Args:
        ui_components: UI components dictionary
        config: Config dictionary (bisa dari inheritance)
    """
    # Extract sections dengan safe defaults (handle inheritance)
    aug_config = config.get('augmentation', {})
    position_config = aug_config.get('position', {})
    lighting_config = aug_config.get('lighting', {})
    
    # Safe update helper dengan validation
    safe_update = lambda key, value: _safe_set_widget_value(ui_components, key, value)
    
    # Basic options mapping dengan safe defaults
    safe_update('num_variations', aug_config.get('num_variations', 3))
    safe_update('target_count', aug_config.get('target_count', 500))
    safe_update('output_prefix', aug_config.get('output_prefix', 'aug'))
    safe_update('balance_classes', aug_config.get('balance_classes', True))
    safe_update('target_split', aug_config.get('target_split', 'train'))
    
    # Augmentation types dengan special handling
    aug_types = aug_config.get('types', ['combined'])
    if isinstance(aug_types, list) and aug_types:
        safe_update('augmentation_types', aug_types)
    
    # Position parameters dengan range validation
    safe_update('fliplr', _validate_range(position_config.get('fliplr', 0.5), 0.0, 1.0))
    safe_update('degrees', _validate_range(position_config.get('degrees', 10), 0, 30))
    safe_update('translate', _validate_range(position_config.get('translate', 0.1), 0.0, 0.25))
    safe_update('scale', _validate_range(position_config.get('scale', 0.1), 0.0, 0.25))
    
    # Lighting parameters dengan range validation
    safe_update('hsv_h', _validate_range(lighting_config.get('hsv_h', 0.015), 0.0, 0.05))
    safe_update('hsv_s', _validate_range(lighting_config.get('hsv_s', 0.7), 0.0, 1.0))
    safe_update('brightness', _validate_range(lighting_config.get('brightness', 0.2), 0.0, 0.4))
    safe_update('contrast', _validate_range(lighting_config.get('contrast', 0.2), 0.0, 0.4))

def reset_augmentation_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke defaults dengan fallback safety"""
    try:
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        default_config = get_default_augmentation_config()
        update_augmentation_ui(ui_components, default_config)
        
        # Log success jika ada logger
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'success'):
            logger.success("🔄 UI augmentasi direset ke default")
            
    except Exception as e:
        # Fallback ke hardcoded defaults
        _apply_hardcoded_defaults(ui_components)
        
        # Log warning jika ada logger
        logger = ui_components.get('logger')
        if logger and hasattr(logger, 'warning'):
            logger.warning(f"⚠️ Error reset, menggunakan fallback: {str(e)}")

def _safe_set_widget_value(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safe set widget value dengan error handling"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            # Special handling untuk SelectMultiple
            if hasattr(widget, 'options') and isinstance(value, list):
                # Validate options exists
                valid_options = [opt[1] if isinstance(opt, tuple) else opt for opt in widget.options]
                filtered_value = [v for v in value if v in valid_options]
                if filtered_value:
                    widget.value = filtered_value
            else:
                widget.value = value
        except Exception:
            # Silent fail - widget mungkin read-only atau invalid value
            pass

def _validate_range(value: float, min_val: float, max_val: float) -> float:
    """Validate value dalam range dengan clamp"""
    try:
        return max(min_val, min(max_val, float(value)))
    except (ValueError, TypeError):
        return min_val

def _apply_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Hardcoded defaults fallback dengan core values"""
    defaults = {
        'num_variations': 3,
        'target_count': 500,
        'output_prefix': 'aug',
        'balance_classes': True,
        'target_split': 'train',
        'augmentation_types': ['combined'],
        'fliplr': 0.5,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.1,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'brightness': 0.2,
        'contrast': 0.2
    }
    
    for key, value in defaults.items():
        _safe_set_widget_value(ui_components, key, value)