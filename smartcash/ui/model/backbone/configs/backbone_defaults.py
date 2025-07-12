"""
File: smartcash/ui/model/backbone/configs/backbone_defaults.py
Default configuration for backbone module following UI module structure standard.
"""

from typing import Dict, Any
from ..constants import DEFAULT_CONFIG, BACKBONE_DEFAULTS, BackboneType

def get_default_backbone_config() -> Dict[str, Any]:
    """
    Get default backbone configuration.
    
    Returns:
        Dict containing default backbone configuration with all required sections
    """
    return {
        'backbone': {
            'model_type': 'efficientnet_b4',
            'pretrained': True,
            'feature_optimization': True,
            'mixed_precision': True,
            'detection_layers': ['banknote'],
            'layer_mode': 'single',
            'input_size': 640,
            'num_classes': 7,
            'early_training': {
                'enabled': True,
                'validation_from_pretrained': True,
                'auto_build': False
            }
        },
        'model': {
            'backbone': 'efficientnet_b4',
            'pretrained': True,
            'detection_layers': ['banknote'],
            'layer_mode': 'single',
            'feature_optimization': True,
            'mixed_precision': True,
            'input_size': 640,
            'num_classes': 7
        },
        'operations': {
            'validate_on_change': True,
            'auto_save_config': True,
            'show_progress': True,
            'timeout_seconds': 300
        },
        'ui': {
            'show_advanced_options': False,
            'auto_validate': True,
            'show_model_info': True,
            'summary_panel_enabled': True
        }
    }

def get_available_backbones() -> Dict[str, Dict[str, Any]]:
    """
    Get available backbone configurations.
    
    Returns:
        Dict containing available backbone configurations
    """
    return {
        'efficientnet_b4': {
            'display_name': 'EfficientNet-B4',
            'description': 'Enhanced accuracy backbone with feature optimization',
            'pretrained_available': True,
            'recommended': True,
            'memory_usage': 'Medium',
            'inference_speed': 'Medium',
            'accuracy': 'High',
            'output_channels': [272, 448, 1792],
            'feature_optimization': True
        },
        'cspdarknet': {
            'display_name': 'CSPDarknet',
            'description': 'YOLOv5 default backbone for fast inference',
            'pretrained_available': True,
            'recommended': False,
            'memory_usage': 'Low',
            'inference_speed': 'Fast',
            'accuracy': 'Medium',
            'output_channels': [256, 512, 1024],
            'feature_optimization': False
        }
    }

def get_detection_layers_config() -> Dict[str, Dict[str, Any]]:
    """
    Get detection layers configuration.
    
    Returns:
        Dict containing detection layers configuration
    """
    return {
        'banknote': {
            'display_name': 'Banknote Detection',
            'description': 'Primary layer for currency banknote detection',
            'classes': ['banknote'],
            'required': True,
            'class_count': 1
        }
    }

def get_layer_modes_config() -> Dict[str, Dict[str, Any]]:
    """
    Get layer modes configuration.
    
    Returns:
        Dict containing layer modes configuration
    """
    return {
        'single': {
            'display_name': 'Single Layer',
            'description': 'Single detection layer for banknote detection only',
            'max_layers': 1,
            'recommended': True,
            'use_case': 'Primary banknote detection'
        }
    }

def get_optimization_config() -> Dict[str, Dict[str, Any]]:
    """
    Get feature optimization configuration.
    
    Returns:
        Dict containing optimization settings
    """
    return {
        'feature_optimization': {
            'enabled': True,
            'use_attention': True,
            'channel_attention': True,
            'spatial_attention': False,
            'testing_mode': False
        },
        'mixed_precision': {
            'enabled': True,
            'fp16': True,
            'auto_scale_loss': True
        },
        'memory_optimization': {
            'gradient_checkpointing': False,
            'pin_memory': True,
            'non_blocking': True
        }
    }