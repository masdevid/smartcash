"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Handler untuk save dan reset configuration dengan train default
"""

from typing import Dict, Any
from smartcash.common.config import get_config_manager

def save_configuration(ui_components: Dict[str, Any]):
    """Save configuration dengan train default"""
    try:
        config = _extract_ui_config(ui_components)
        config_manager = get_config_manager()
        
        success = config_manager.save_config(config, 'augmentation_config.yaml')
        message = '✅ Konfigurasi berhasil disimpan' if success else '❌ Gagal menyimpan konfigurasi'
        
        _log_to_ui(ui_components, message, 'success' if success else 'error')
        
    except Exception as e:
        _log_to_ui(ui_components, f'❌ Error save config: {str(e)}', 'error')

def reset_configuration(ui_components: Dict[str, Any]):
    """Reset configuration ke default dengan train default"""
    try:
        default_config = _get_default_config()
        _apply_config_to_ui(ui_components, default_config)
        
        # Save default config
        config_manager = get_config_manager()
        config_manager.save_config(default_config, 'augmentation_config.yaml')
        
        _log_to_ui(ui_components, '🔄 Konfigurasi direset ke default', 'success')
        
    except Exception as e:
        _log_to_ui(ui_components, f'❌ Error reset config: {str(e)}', 'error')

def _extract_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI dengan train enforcement"""
    # Get augmentation types
    aug_types_widget = ui_components.get('augmentation_types')
    aug_types = list(aug_types_widget.value) if aug_types_widget and hasattr(aug_types_widget, 'value') and aug_types_widget.value else ['combined']
    
    return {
        'augmentation': {
            'num_variations': _get_widget_value(ui_components, 'num_variations', 2),
            'target_count': _get_widget_value(ui_components, 'target_count', 500),
            'output_prefix': _get_widget_value(ui_components, 'output_prefix', 'aug'),
            'balance_classes': _get_widget_value(ui_components, 'balance_classes', False),
            'fliplr': _get_widget_value(ui_components, 'fliplr', 0.5),
            'degrees': _get_widget_value(ui_components, 'degrees', 10),
            'translate': _get_widget_value(ui_components, 'translate', 0.1),
            'scale': _get_widget_value(ui_components, 'scale', 0.1),
            'hsv_h': _get_widget_value(ui_components, 'hsv_h', 0.015),
            'hsv_s': _get_widget_value(ui_components, 'hsv_s', 0.7),
            'brightness': _get_widget_value(ui_components, 'brightness', 0.2),
            'contrast': _get_widget_value(ui_components, 'contrast', 0.2),
            'types': aug_types,
            'target_split': _get_widget_value(ui_components, 'target_split', 'train'),
            'output_dir': 'data/augmented'
        },
        'data': {'dir': 'data'},
        'preprocessing': {'output_dir': 'data/preprocessed'}
    }

def _apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Apply config ke UI widgets dengan train enforcement"""
    aug_config = config.get('augmentation', {})
    
    # Apply basic values
    widget_mappings = {
        'num_variations': aug_config.get('num_variations', 2),
        'target_count': aug_config.get('target_count', 500),
        'output_prefix': aug_config.get('output_prefix', 'aug'),
        'balance_classes': aug_config.get('balance_classes', False),
        'fliplr': aug_config.get('fliplr', 0.5),
        'degrees': aug_config.get('degrees', 10),
        'translate': aug_config.get('translate', 0.1),
        'scale': aug_config.get('scale', 0.1),
        'hsv_h': aug_config.get('hsv_h', 0.015),
        'hsv_s': aug_config.get('hsv_s', 0.7),
        'brightness': aug_config.get('brightness', 0.2),
        'contrast': aug_config.get('contrast', 0.2)
    }
    
    for widget_key, value in widget_mappings.items():
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            widget.value = value
    
    # Apply augmentation types dan target split
    aug_types_widget = ui_components.get('augmentation_types')
    if aug_types_widget and hasattr(aug_types_widget, 'value'):
        aug_types_widget.value = list(aug_config.get('types', ['combined']))
    
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        target_split_widget.value = aug_config.get('target_split', 'train')

def _get_default_config() -> Dict[str, Any]:
    """Default config dengan train default"""
    return {
        'augmentation': {
            'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug', 'balance_classes': False,
            'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2,
            'types': ['combined'], 'target_split': 'train', 'intensity': 0.7, 'output_dir': 'data/augmented'
        },
        'data': {'dir': 'data'},
        'preprocessing': {'output_dir': 'data/preprocessed'}
    }

# One-liner utils
_get_widget_value = lambda ui_components, key, default: getattr(ui_components.get(key), 'value', default)

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message ke UI"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)