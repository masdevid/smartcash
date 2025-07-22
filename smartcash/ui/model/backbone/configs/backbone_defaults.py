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
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],  # Multi-layer by default
            'layer_mode': 'multi',  # Multi-layer mode by default
            'multi_layer_heads': True,  # Enable multi-layer heads
            'save_path': 'data/models',  # Default model save path
            'input_size': 640,  # Fixed as per MODEL_ARC_README.md
            'num_classes': {  # Fixed as per MODEL_ARC_README.md
                'layer_1': 7,   # 7 denominations
                'layer_2': 7,   # 7 denomination-specific features  
                'layer_3': 3    # 3 common features
            },
            'early_training': {
                'enabled': True,
                'validation_from_pretrained': True,
                'auto_build': False
            }
        },
        'model': {
            'backbone': 'efficientnet_b4',
            'pretrained': True,
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],  # Multi-layer by default
            'layer_mode': 'multi',  # Multi-layer mode by default
            'multi_layer_heads': True,  # Enable multi-layer heads
            'save_path': 'data/models',  # Default model save path
            'feature_optimization': True,
            'mixed_precision': True,
            'input_size': 640,  # Fixed as per MODEL_ARC_README.md
            'num_classes': {  # Fixed as per MODEL_ARC_README.md
                'layer_1': 7,   # 7 denominations
                'layer_2': 7,   # 7 denomination-specific features  
                'layer_3': 3    # 3 common features
            }
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
    Get detection layers configuration based on MODEL_ARC_README.md.
    
    Returns:
        Dict containing detection layers configuration
    """
    return {
        'layer_1': {
            'display_name': 'Layer 1: Full Banknote Detection',
            'description': 'Detects full note bounding boxes',
            'classes': ['001', '002', '005', '010', '020', '050', '100'],
            'class_names': ['1K IDR', '2K IDR', '5K IDR', '10K IDR', '20K IDR', '50K IDR', '100K IDR'],
            'required': True,
            'class_count': 7,
            'examples': ['100K IDR', '50K IDR'],
            'purpose': 'Full banknote detection (main object)'
        },
        'layer_2': {
            'display_name': 'Layer 2: Denomination Features',
            'description': 'Detects denomination-specific visual markers',
            'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
            'class_names': ['1K Features', '2K Features', '5K Features', '10K Features', '20K Features', '50K Features', '100K Features'],
            'required': True,
            'class_count': 7,
            'examples': ['Large printed number', 'Portrait', 'Watermark', 'Braile'],
            'purpose': 'Nominal-defining features (unique visual cues)'
        },
        'layer_3': {
            'display_name': 'Layer 3: Common Features',
            'description': 'Detects common features across all notes',
            'classes': ['l3_sign', 'l3_text', 'l3_thread'],
            'class_names': ['BI Logo', 'Serial Number & Micro Text', 'Security Thread'],
            'required': True,
            'class_count': 3,
            'examples': ['BI Logo', 'Serial Number & Micro Text', 'Security Thread'],
            'purpose': 'Common features (shared among notes)'
        }
    }

def get_layer_modes_config() -> Dict[str, Dict[str, Any]]:
    """
    Get layer modes configuration based on MODEL_ARC_README.md.
    
    Returns:
        Dict containing layer modes configuration
    """
    return {
        'multi': {
            'display_name': 'Multi-Layer Detection',
            'description': '3-layer detection system for comprehensive banknote analysis',
            'max_layers': 3,
            'recommended': True,
            'use_case': 'Full banknote detection with denomination and feature analysis',
            'layers': ['layer_1', 'layer_2', 'layer_3'],
            'total_classes': 17,  # 7 + 7 + 3
            'training_strategy': 'Two-phase with uncertainty-based multi-task loss'
        },
        'single': {
            'display_name': 'Single Layer (Legacy)',
            'description': 'Single detection layer for basic banknote detection only',
            'max_layers': 1,
            'recommended': False,
            'use_case': 'Basic banknote detection (legacy mode)',
            'layers': ['layer_1'],
            'total_classes': 7,
            'training_strategy': 'Standard YOLO training'
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