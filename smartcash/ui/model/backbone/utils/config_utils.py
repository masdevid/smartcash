"""
File: smartcash/ui/model/backbone/utils/config_utils.py
Deskripsi: Configuration utilities untuk backbone model
"""

from typing import Dict, Any, List, Optional
import copy

def extract_essential_config(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only essential configuration fields
    
    Args:
        full_config: Full configuration dictionary
        
    Returns:
        Essential configuration only
    """
    if 'model' not in full_config:
        return {'model': get_minimal_config()['model']}
    
    model_config = full_config['model']
    
    essential = {
        'model': {
            'backbone': model_config.get('backbone', 'efficientnet_b4'),
            'model_name': model_config.get('model_name', 'smartcash_yolov5'),
            'detection_layers': model_config.get('detection_layers', ['banknote']),
            'layer_mode': model_config.get('layer_mode', 'single'),
            'num_classes': 7,  # Fixed for SmartCash
            'img_size': 640,   # Fixed for consistency
            'feature_optimization': {
                'enabled': model_config.get('feature_optimization', {}).get('enabled', False)
            },
            'mixed_precision': model_config.get('mixed_precision', True),
            'device': 'auto'  # Always auto-detect
        }
    }
    
    return essential

def get_minimal_config() -> Dict[str, Any]:
    """Get minimal valid configuration
    
    Returns:
        Minimal configuration dictionary
    """
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

def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration with defaults
    
    Args:
        config: Partial configuration
        
    Returns:
        Complete configuration with defaults
    """
    defaults = get_minimal_config()
    return deep_merge(defaults, config)

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result

def config_to_api_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI config to API parameters
    
    Args:
        config: UI configuration
        
    Returns:
        API-compatible parameters
    """
    model_config = config.get('model', {})
    
    api_params = {
        'backbone': model_config.get('backbone', 'efficientnet_b4'),
        'detection_layers': model_config.get('detection_layers', ['banknote']),
        'layer_mode': model_config.get('layer_mode', 'single'),
        'num_classes': model_config.get('num_classes', 7),
        'img_size': model_config.get('img_size', 640),
        'feature_optimization': model_config.get('feature_optimization', {'enabled': False}),
        'device': model_config.get('device', 'auto')
    }
    
    # Add runtime parameters if present
    if 'inference' in config:
        api_params.update({
            'confidence_threshold': config['inference'].get('confidence_threshold', 0.25),
            'iou_threshold': config['inference'].get('iou_threshold', 0.45),
            'max_detections': config['inference'].get('max_detections', 100)
        })
    
    return api_params

def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two configurations and return differences
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary of differences
    """
    differences = {}
    
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        if key not in config1:
            differences[key] = {'added': config2[key]}
        elif key not in config2:
            differences[key] = {'removed': config1[key]}
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            sub_diff = compare_configs(config1[key], config2[key])
            if sub_diff:
                differences[key] = sub_diff
        elif config1[key] != config2[key]:
            differences[key] = {
                'old': config1[key],
                'new': config2[key]
            }
    
    return differences

def format_config_summary(config: Dict[str, Any]) -> List[str]:
    """Format configuration for display
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of formatted strings
    """
    model_config = config.get('model', {})
    
    summary = []
    summary.append(f"Backbone: {model_config.get('backbone', 'N/A')}")
    summary.append(f"Detection Layers: {', '.join(model_config.get('detection_layers', []))}")
    summary.append(f"Layer Mode: {model_config.get('layer_mode', 'N/A')}")
    summary.append(f"Feature Optimization: {'Enabled' if model_config.get('feature_optimization', {}).get('enabled') else 'Disabled'}")
    summary.append(f"Mixed Precision: {'FP16' if model_config.get('mixed_precision') else 'FP32'}")
    summary.append(f"Input Size: {model_config.get('img_size', 640)}x{model_config.get('img_size', 640)}")
    summary.append(f"Classes: {model_config.get('num_classes', 7)}")
    
    return summary