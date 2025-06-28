"""
File: smartcash/ui/model/backbone/utils/ui_utils.py
Deskripsi: UI utilities untuk backbone model configuration
"""

from typing import Dict, Any, Optional

def extract_model_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration dari UI components"""
    # Helper untuk safe value extraction
    get_value = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
    
    # Extract form values
    backbone = get_value('backbone_dropdown', 'efficientnet_b4')
    detection_layers = list(get_value('detection_layers_select', ['banknote']))
    layer_mode = get_value('layer_mode_dropdown', 'single')
    feature_optimization = get_value('feature_optimization_checkbox', False)
    mixed_precision = get_value('mixed_precision_checkbox', True)
    
    # Build config structure
    config = {
        'model': {
            'backbone': backbone,
            'model_name': 'smartcash_yolov5',
            'detection_layers': detection_layers,
            'layer_mode': layer_mode,
            'num_classes': 7,  # Fixed untuk SmartCash
            'img_size': 640,   # Fixed untuk consistency
            'feature_optimization': {
                'enabled': feature_optimization
            },
            'mixed_precision': mixed_precision,
            'device': 'auto'  # Auto-detect
        }
    }
    
    return config

def update_model_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dengan configuration values"""
    model_config = config.get('model', {})
    
    # Helper untuk safe update
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update form values
    safe_update('backbone_dropdown', model_config.get('backbone', 'efficientnet_b4'))
    safe_update('detection_layers_select', tuple(model_config.get('detection_layers', ['banknote'])))
    safe_update('layer_mode_dropdown', model_config.get('layer_mode', 'single'))
    safe_update('feature_optimization_checkbox', model_config.get('feature_optimization', {}).get('enabled', False))
    safe_update('mixed_precision_checkbox', model_config.get('mixed_precision', True))
    
    # Update config summary if available
    if 'config_summary' in ui_components:
        from smartcash.ui.model.backbone.components.config_summary import update_config_summary
        update_config_summary(ui_components['config_summary'], config)

def reset_model_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default values"""
    default_config = get_default_model_config()
    update_model_ui(ui_components, default_config)

def get_default_model_config() -> Dict[str, Any]:
    """Get default model configuration"""
    return {
        'model': {
            'backbone': 'efficientnet_b4',
            'model_name': 'smartcash_yolov5',
            'detection_layers': ['banknote'],
            'layer_mode': 'single',
            'num_classes': 7,
            'img_size': 640,
            'feature_optimization': {
                'enabled': False
            },
            'mixed_precision': True,
            'device': 'auto'
        }
    }

def validate_model_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """Validate model configuration"""
    model_config = config.get('model', {})
    
    # Check required fields
    if not model_config.get('backbone'):
        return False, "Backbone harus dipilih"
    
    if not model_config.get('detection_layers'):
        return False, "Minimal satu detection layer harus dipilih"
    
    # Validate backbone value
    valid_backbones = ['efficientnet_b4', 'cspdarknet']
    if model_config.get('backbone') not in valid_backbones:
        return False, f"Invalid backbone: {model_config.get('backbone')}"
    
    # Validate layer mode
    valid_modes = ['single', 'multilayer']
    if model_config.get('layer_mode') not in valid_modes:
        return False, f"Invalid layer mode: {model_config.get('layer_mode')}"
    
    return True, "Configuration valid"