"""
File: smartcash/ui/model/backbone/constants.py
Constants for backbone models module following UI module structure standard.
"""

from enum import Enum
from typing import Dict, List, Any

# ==================== Enums ====================

class BackboneType(Enum):
    """Available backbone types for model configuration."""
    CSPDARKNET = "cspdarknet"
    EFFICIENTNET_B4 = "efficientnet_b4"

class BackboneOperation(Enum):
    """Operations available in backbone module."""
    VALIDATE = "validate"
    BUILD = "build"

# ==================== Default Configurations ====================

BACKBONE_DEFAULTS = {
    BackboneType.CSPDARKNET.value: {
        'name': 'CSPDarknet',
        'description': 'YOLOv5 default CSPDarknet backbone for fast inference',
        'pretrained': True,
        'feature_optimization': False,
        'memory_usage': 'Low',
        'inference_speed': 'Fast',
        'output_channels': [256, 512, 1024]
    },
    BackboneType.EFFICIENTNET_B4.value: {
        'name': 'EfficientNet-B4',
        'description': 'EfficientNet-B4 backbone for enhanced accuracy',
        'pretrained': True,
        'feature_optimization': True,
        'memory_usage': 'Medium',
        'inference_speed': 'Medium',
        'output_channels': [272, 448, 1792]
    }
}

# ==================== Progress Steps ====================

PROGRESS_STEPS = {
    BackboneOperation.VALIDATE.value: [
        "🔍 Validating backbone configuration",
        "📋 Checking model compatibility", 
        "✅ Validation complete"
    ],
    BackboneOperation.BUILD.value: [
        "🏗️ Building backbone architecture",
        "📥 Auto-loading pretrained weights",
        "🔧 Configuring model layers",
        "📊 Calculating parameters",
        "✅ Build complete"
    ]
}

# ==================== UI Configuration ====================

UI_CONFIG = {
    'module_name': 'backbone',
    'parent_module': 'model',
    'title': 'Model Backbone',
    'subtitle': 'Configure backbone architecture for feature extraction',
    'description': 'Select and configure the backbone model for currency detection',
    'icon': '🧬',
    'version': '2.0.0'
}

# Module Metadata
MODULE_METADATA = {
    'name': 'backbone',
    'title': 'Model Backbone',
    'description': 'Backbone model configuration module with early training pipeline integration',
    'version': '2.0.0',
    'category': 'model',
    'author': 'SmartCash',
    'tags': ['backbone', 'model', 'cspdarknet', 'efficientnet', 'configuration'],
    'features': [
        'CSPDarknet backbone support',
        'EfficientNet-B4 backbone support',
        'Model builder integration',
        'Configuration validation',
        'Early training pipeline',
        'Backend model integration'
    ]
}

# Button Configuration
BUTTON_CONFIG = {
    'validate': {
        'text': '🔍 Validate',
        'style': 'info',
        'tooltip': 'Validate backbone configuration and compatibility',
        'order': 1
    },
    'build': {
        'text': '🏗️ Build Model',
        'style': 'success',
        'tooltip': 'Build backbone architecture with current configuration (pretrained auto-loaded from drive)',
        'order': 2
    },
    'rescan_models': {
        'text': '🔄 Rescan Models',
        'style': 'info',
        'tooltip': 'Rescan for existing built models',
        'order': 3
    }
}

# ==================== Configuration Options ====================

DETECTION_LAYERS = {
    'banknote': {
        'display_name': 'Banknote Detection',
        'description': 'Primary layer for banknote detection',
        'required': True,
        'classes': 1
    }
}

LAYER_MODES = {
    'single': {
        'display_name': 'Single Layer',
        'description': 'Single detection layer (banknote only)',
        'recommended': True
    }
}

# ==================== Model Configuration ====================

DEFAULT_MODEL_CONFIG = {
    'backbone': 'efficientnet_b4',
    'pretrained': True,
    'detection_layers': ['banknote'],
    'layer_mode': 'single',
    'feature_optimization': True,
    'mixed_precision': True,
    'input_size': 640,
    'num_classes': 7
}

# ==================== Validation Settings ====================

VALIDATION_CONFIG = {
    'required_fields': ['backbone', 'pretrained', 'detection_layers'],
    'allowed_backbones': list(BackboneType.__members__.keys()).copy(),
    'min_input_size': 320,
    'max_input_size': 1280,
    'min_classes': 1,
    'max_classes': 100
}

# ==================== Messages ====================

ERROR_MESSAGES = {
    'invalid_backbone': "Invalid backbone type selected",
    'loading_failed': "Failed to load backbone model",
    'build_failed': "Failed to build backbone architecture",
    'validation_failed': "Backbone configuration validation failed",
    'backend_error': "Backend service error occurred"
}

SUCCESS_MESSAGES = {
    'validation_success': "Backbone configuration validated successfully",
    'load_success': "Backbone model loaded successfully",
    'build_success': "Backbone architecture built successfully",
    'summary_success': "Model summary generated successfully"
}

# ==================== Defaults ====================

DEFAULT_CONFIG = {
    'model': DEFAULT_MODEL_CONFIG.copy(),
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