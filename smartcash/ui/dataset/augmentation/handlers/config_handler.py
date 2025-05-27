"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Config handler dengan unified logging dan parameter alignment
"""

from typing import Dict, Any
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui

def save_configuration(ui_components: Dict[str, Any]):
    """Save config dengan parameter alignment"""
    try:
        config = _extract_ui_config(ui_components)
        
        if not _validate_config(config):
            log_to_ui(ui_components, 'Config tidak valid - periksa parameter', 'error', '❌ ')
            return
        
        config_manager = get_config_manager()
        success = config_manager.save_config(config, 'augmentation_config.yaml')
        
        if success:
            _update_cache(ui_components, config)
            log_to_ui(ui_components, 'Config berhasil disimpan dengan parameter alignment', 'success', '✅ ')
        else:
            log_to_ui(ui_components, 'Gagal menyimpan config', 'error', '❌ ')
        
    except Exception as e:
        log_to_ui(ui_components, f'Config save error: {str(e)}', 'error', '❌ ')

def reset_configuration(ui_components: Dict[str, Any]):
    """Reset config dengan research defaults"""
    try:
        default_config = _get_default_config()
        _apply_config_to_ui(ui_components, default_config)
        
        config_manager = get_config_manager()
        config_manager.save_config(default_config, 'augmentation_config.yaml')
        _update_cache(ui_components, default_config)
        
        log_to_ui(ui_components, 'Config direset ke research defaults', 'success', '🔄 ')
        
    except Exception as e:
        log_to_ui(ui_components, f'Config reset error: {str(e)}', 'error', '❌ ')

def _extract_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dengan parameter alignment"""
    try:
        from smartcash.dataset.augmentor.config import extract_ui_config
        return extract_ui_config(ui_components)
    except ImportError:
        return _manual_extraction(ui_components)

def _manual_extraction(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Manual extraction dengan research parameters"""
    from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
    
    aug_types = _extract_aug_types(ui_components)
    basic_params = {
        'num_variations': _get_widget_value_safe(ui_components, 'num_variations', 2),
        'target_count': _get_widget_value_safe(ui_components, 'target_count', 500),
        'target_split': _get_widget_value_safe(ui_components, 'target_split', 'train'),
        'output_prefix': _get_widget_value_safe(ui_components, 'output_prefix', 'aug_'),
        'balance_classes': _get_widget_value_safe(ui_components, 'balance_classes', False)
    }
    
    advanced_params = {
        'fliplr': _get_widget_value_safe(ui_components, 'fliplr', 0.5),
        'degrees': _get_widget_value_safe(ui_components, 'degrees', 10),
        'translate': _get_widget_value_safe(ui_components, 'translate', 0.1),
        'scale': _get_widget_value_safe(ui_components, 'scale', 0.1),
        'hsv_h': _get_widget_value_safe(ui_components, 'hsv_h', 0.015),
        'hsv_s': _get_widget_value_safe(ui_components, 'hsv_s', 0.7),
        'brightness': _get_widget_value_safe(ui_components, 'brightness', 0.2),
        'contrast': _get_widget_value_safe(ui_components, 'contrast', 0.2)
    }
    
    return {
        'data': {'dir': get_best_data_location()},
        'augmentation': {'types': aug_types, 'intensity': 0.7, 'output_dir': 'data/augmented', **basic_params, **advanced_params},
        'preprocessing': {'output_dir': 'data/preprocessed'}
    }

def _extract_aug_types(ui_components: Dict[str, Any]) -> list:
    """Extract augmentation types dengan fallback strategies"""
    widget = ui_components.get('augmentation_types')
    if widget and hasattr(widget, 'value') and widget.value:
        return list(widget.value)
    
    for alt_name in ['types_widget', 'aug_types', 'augmentation_type']:
        widget = ui_components.get(alt_name)
        if widget and hasattr(widget, 'value') and widget.value:
            return list(widget.value)
    
    return ['combined']  # Research default

def _validate_config(config: Dict[str, Any]) -> bool:
    """Validate config untuk research compatibility"""
    try:
        aug_config = config.get('augmentation', {})
        
        if aug_config.get('num_variations', 0) <= 0:
            return False
        if aug_config.get('target_count', 0) <= 0:
            return False
        if not aug_config.get('types'):
            return False
        
        ranges = {
            'fliplr': (0.0, 1.0), 'degrees': (0, 45), 'translate': (0.0, 0.5), 'scale': (0.0, 0.5),
            'hsv_h': (0.0, 0.1), 'hsv_s': (0.0, 1.0), 'brightness': (0.0, 1.0), 'contrast': (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in ranges.items():
            value = aug_config.get(param)
            if value is not None and not (min_val <= value <= max_val):
                return False
        
        return True
        
    except Exception:
        return False

def _get_default_config() -> Dict[str, Any]:
    """Default config dengan research optimization"""
    return {
        'data': {'dir': 'data'},
        'augmentation': {
            'types': ['combined'], 'num_variations': 2, 'target_count': 500, 'target_split': 'train',
            'intensity': 0.7, 'output_prefix': 'aug_', 'balance_classes': False, 'output_dir': 'data/augmented',
            'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2
        },
        'preprocessing': {'output_dir': 'data/preprocessed'}
    }

def _apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Apply config ke UI dengan parameter alignment"""
    aug_config = config.get('augmentation', {})
    
    basic_mappings = {
        'num_variations': aug_config.get('num_variations', 2),
        'target_count': aug_config.get('target_count', 500),
        'target_split': aug_config.get('target_split', 'train'),
        'output_prefix': aug_config.get('output_prefix', 'aug_'),
        'balance_classes': aug_config.get('balance_classes', False)
    }
    
    advanced_mappings = {
        'fliplr': aug_config.get('fliplr', 0.5), 'degrees': aug_config.get('degrees', 10),
        'translate': aug_config.get('translate', 0.1), 'scale': aug_config.get('scale', 0.1),
        'hsv_h': aug_config.get('hsv_h', 0.015), 'hsv_s': aug_config.get('hsv_s', 0.7),
        'brightness': aug_config.get('brightness', 0.2), 'contrast': aug_config.get('contrast', 0.2)
    }
    
    all_mappings = {**basic_mappings, **advanced_mappings}
    for widget_key, value in all_mappings.items():
        _set_widget_value_safe(ui_components, widget_key, value)
    
    aug_types = aug_config.get('types', ['combined'])
    _set_augmentation_types(ui_components, aug_types)

def _update_cache(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update cache dengan new config"""
    ui_components['config'] = config
    ui_components['config_cache_valid'] = True

def _set_augmentation_types(ui_components: Dict[str, Any], types: list) -> None:
    """Set augmentation types dengan multiple strategies"""
    for widget_key in ['augmentation_types', 'types_widget', 'aug_types']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = list(types)
                return
            except Exception:
                continue

def _get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Safe widget value extraction dengan type consistency"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = widget.value
            if isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            return value
        except Exception:
            pass
    return default

def _set_widget_value_safe(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safe widget value setting"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            widget.value = value
        except Exception:
            pass