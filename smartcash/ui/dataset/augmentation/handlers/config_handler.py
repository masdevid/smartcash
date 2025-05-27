"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Fixed config handler dengan parameter alignment dan cache management
"""

from typing import Dict, Any
from smartcash.common.config import get_config_manager

def save_configuration(ui_components: Dict[str, Any]):
    """Save configuration dengan aligned parameters"""
    try:
        # Extract config dengan parameter alignment
        config = _extract_aligned_ui_config(ui_components)
        
        # Validate config sebelum save
        if not _validate_config_parameters(config):
            _log_to_ui(ui_components, '❌ Konfigurasi tidak valid - periksa parameter', 'error')
            return
        
        # Save dengan config manager
        config_manager = get_config_manager()
        success = config_manager.save_config(config, 'augmentation_config.yaml')
        
        if success:
            # Update cache dengan new config
            _update_component_cache(ui_components, config)
            
            _log_to_ui(ui_components, '✅ Konfigurasi berhasil disimpan dan cache diperbarui', 'success')
        else:
            _log_to_ui(ui_components, '❌ Gagal menyimpan konfigurasi', 'error')
        
    except Exception as e:
        _log_to_ui(ui_components, f'❌ Error save config: {str(e)}', 'error')

def reset_configuration(ui_components: Dict[str, Any]):
    """Reset configuration dengan aligned parameters dan cache invalidation"""
    try:
        # Get default config dengan research-friendly parameters
        default_config = _get_default_config()
        
        # Apply config ke UI widgets
        _apply_aligned_config_to_ui(ui_components, default_config)
        
        # Save default config
        config_manager = get_config_manager()
        config_manager.save_config(default_config, 'augmentation_config.yaml')
        
        # Invalidate cache untuk force refresh
        _invalidate_component_cache(ui_components)
        
        _log_to_ui(ui_components, '🔄 Konfigurasi direset ke research-friendly defaults', 'success')
        
    except Exception as e:
        _log_to_ui(ui_components, f'❌ Error reset config: {str(e)}', 'error')

def _extract_aligned_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dengan parameter alignment yang comprehensive"""
    try:
        from smartcash.dataset.augmentor.config import extract_ui_config
        return extract_ui_config(ui_components)
    except ImportError:
        # Fallback manual extraction
        return _manual_config_extraction(ui_components)

def _manual_config_extraction(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Manual config extraction sebagai fallback"""
    # Extract augmentation types dengan multiple strategies
    aug_types = _extract_aug_types_comprehensive(ui_components)
    
    # Extract basic parameters
    basic_params = {
        'num_variations': _get_widget_value_safe(ui_components, 'num_variations', 2),
        'target_count': _get_widget_value_safe(ui_components, 'target_count', 500),
        'target_split': _get_widget_value_safe(ui_components, 'target_split', 'train'),
        'output_prefix': _get_widget_value_safe(ui_components, 'output_prefix', 'aug_'),
        'balance_classes': _get_widget_value_safe(ui_components, 'balance_classes', False)
    }
    
    # Extract advanced parameters (UI alignment)
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
        'data': {'dir': 'data'},
        'augmentation': {
            'types': aug_types,
            'intensity': 0.7,
            'output_dir': 'data/augmented',
            **basic_params,
            **advanced_params
        },
        'preprocessing': {'output_dir': 'data/preprocessed'}
    }

def _extract_aug_types_comprehensive(ui_components: Dict[str, Any]) -> list:
    """Comprehensive augmentation types extraction"""
    # Strategy 1: Primary widget
    widget = ui_components.get('augmentation_types')
    if widget and hasattr(widget, 'value') and widget.value:
        return list(widget.value)
    
    # Strategy 2: Alternative names
    for alt_name in ['types_widget', 'aug_types', 'augmentation_type']:
        widget = ui_components.get(alt_name)
        if widget and hasattr(widget, 'value') and widget.value:
            return list(widget.value)
    
    # Strategy 3: Container approach
    aug_options = ui_components.get('aug_options')
    if aug_options and hasattr(aug_options, 'children'):
        for child in aug_options.children:
            if hasattr(child, 'value') and child.value:
                return list(child.value)
    
    # Default fallback
    return ['combined']

def _apply_aligned_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Apply config ke UI dengan parameter alignment"""
    aug_config = config.get('augmentation', {})
    
    # Basic parameters
    basic_mappings = {
        'num_variations': aug_config.get('num_variations', 2),
        'target_count': aug_config.get('target_count', 500),
        'target_split': aug_config.get('target_split', 'train'),
        'output_prefix': aug_config.get('output_prefix', 'aug_'),
        'balance_classes': aug_config.get('balance_classes', False)
    }
    
    # Advanced parameters (UI alignment)
    advanced_mappings = {
        'fliplr': aug_config.get('fliplr', 0.5),
        'degrees': aug_config.get('degrees', 10),
        'translate': aug_config.get('translate', 0.1),
        'scale': aug_config.get('scale', 0.1),
        'hsv_h': aug_config.get('hsv_h', 0.015),
        'hsv_s': aug_config.get('hsv_s', 0.7),
        'brightness': aug_config.get('brightness', 0.2),
        'contrast': aug_config.get('contrast', 0.2)
    }
    
    # Apply all mappings dengan error handling
    all_mappings = {**basic_mappings, **advanced_mappings}
    for widget_key, value in all_mappings.items():
        _set_widget_value_robust(ui_components, widget_key, value)
    
    # Apply augmentation types
    aug_types = aug_config.get('types', ['combined'])
    _set_augmentation_types_robust(ui_components, aug_types)

def _validate_config_parameters(config: Dict[str, Any]) -> bool:
    """Validate config parameters untuk ensure correctness"""
    try:
        aug_config = config.get('augmentation', {})
        
        # Basic validations
        if aug_config.get('num_variations', 0) <= 0:
            return False
        if aug_config.get('target_count', 0) <= 0:
            return False
        if not aug_config.get('types'):
            return False
        
        # Range validations untuk UI parameters
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
    """Default config dengan research-friendly parameters"""
    return {
        'data': {'dir': 'data'},
        'augmentation': {
            # Research pipeline defaults
            'types': ['combined'],  # Combined position + lighting
            'num_variations': 2,
            'target_count': 500,
            'target_split': 'train',
            'intensity': 0.7,
            'output_prefix': 'aug_',
            'balance_classes': False,
            'output_dir': 'data/augmented',
            
            # Position parameters (moderate untuk currency)
            'fliplr': 0.5,
            'degrees': 10,      # Conservative rotation
            'translate': 0.1,   # Minimal translation
            'scale': 0.1,       # Minimal scaling
            
            # Lighting parameters (research-optimal)
            'hsv_h': 0.015,     # Minimal hue shift
            'hsv_s': 0.7,       # Moderate saturation
            'brightness': 0.2,  # Moderate brightness
            'contrast': 0.2     # Moderate contrast
        },
        'preprocessing': {'output_dir': 'data/preprocessed'}
    }

def _update_component_cache(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update component cache dengan new config"""
    ui_components['config'] = config
    ui_components['last_config_hash'] = hash(str(config))
    ui_components['config_cache_valid'] = True

def _invalidate_component_cache(ui_components: Dict[str, Any]) -> None:
    """Invalidate component cache untuk force refresh"""
    ui_components['config_cache_valid'] = False
    ui_components.pop('last_config_hash', None)

# Safe widget operations
def _get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Safe widget value extraction dengan type consistency"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = widget.value
            # Type consistency
            if isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            return value
        except Exception:
            pass
    return default

def _set_widget_value_robust(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Robust widget value setting dengan error handling"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            widget.value = value
        except Exception:
            pass  # Silent fail untuk widget compatibility

def _set_augmentation_types_robust(ui_components: Dict[str, Any], types: list) -> None:
    """Robust augmentation types setting dengan multiple strategies"""
    # Strategy 1: Primary widget
    for widget_key in ['augmentation_types', 'types_widget', 'aug_types']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = list(types)
                return
            except Exception:
                continue
    
    # Strategy 2: Container approach
    aug_options = ui_components.get('aug_options')
    if aug_options and hasattr(aug_options, 'children'):
        for child in aug_options.children:
            if hasattr(child, 'value'):
                try:
                    child.value = list(types)
                    return
                except Exception:
                    continue

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message ke UI dengan safe error handling"""
    try:
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Fallback ke widget display
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            from IPython.display import display, HTML
            color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = color_map.get(level, '#007bff')
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
            
            with widget:
                display(HTML(html))
    except Exception:
        pass  # Silent fail untuk prevent error chains