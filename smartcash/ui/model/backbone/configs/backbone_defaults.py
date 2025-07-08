"""
File: smartcash/ui/model/backbone/configs/backbone_defaults.py
Description: Default configuration for backbone module following UI module structure standard
"""

from typing import Dict, Any
from ..constants import BACKBONE_DEFAULTS, BackboneType, CONFIG_KEYS

def get_default_backbone_config() -> Dict[str, Any]:
    """
    Get default backbone configuration.
    
    Returns:
        Dict containing default backbone configuration
    """
    return {
        'model': {
            'backbone': 'efficientnet_b4',
            'pretrained': True,
            'detection_layers': ['banknote'],
            'layer_mode': 'single',
            'feature_optimization': {
                'enabled': True,
                'use_attention': True,
                'testing_mode': False
            },
            'mixed_precision': True,
            'input_size': 640,
            'num_classes': 7
        },
        'ui': {
            'show_advanced_options': False,
            'auto_validate': True,
            'show_model_info': True
        },
        'validation': {
            'auto_validate_on_change': True,
            'show_compatibility_warnings': True
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
            'description': 'EfficientNet-B4 backbone for enhanced accuracy',
            'pretrained_available': True,
            'recommended_for': 'High-accuracy currency detection',
            'memory_usage': 'Medium',
            'inference_speed': 'Medium',
            'feature_maps': 3,
            'output_channels': [272, 448, 1792]
        },
        'cspdarknet': {
            'display_name': 'CSPDarknet',
            'description': 'YOLOv5 default CSPDarknet backbone',
            'pretrained_available': True,
            'recommended_for': 'General object detection',
            'memory_usage': 'Low',
            'inference_speed': 'Fast',
            'feature_maps': 3,
            'output_channels': [256, 512, 1024]
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
            'description': 'Primary layer for banknote detection',
            'classes': ['banknote'],
            'required': True
        },
        'nominal': {
            'display_name': 'Nominal Area',
            'description': 'Secondary layer for nominal value detection',
            'classes': ['1000', '2000', '5000', '10000', '20000', '50000', '100000'],
            'required': False
        },
        'security': {
            'display_name': 'Security Features',
            'description': 'Tertiary layer for security features',
            'classes': ['watermark', 'security_thread', 'microtext'],
            'required': False
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
            'description': 'Single detection layer (banknote only)',
            'max_layers': 1,
            'recommended_for': 'Simple banknote detection'
        },
        'multilayer': {
            'display_name': 'Multi-Layer',
            'description': 'Multiple detection layers (banknote + nominal + security)',
            'max_layers': 3,
            'recommended_for': 'Comprehensive currency analysis'
        }
    }