"""
File: smartcash/ui/model/backbone/constants.py
Description: Constants for backbone module following UI module structure standard
"""

from enum import Enum
from typing import Dict, List, Any

# Backbone Types
class BackboneType(Enum):
    """Available backbone architectures"""
    CSPDARKNET = "cspdarknet"
    EFFICIENTNET_B4 = "efficientnet_b4"

# Default Backbone Configurations
BACKBONE_DEFAULTS = {
    BackboneType.CSPDARKNET.value: {
        'name': 'CSPDarknet',
        'description': 'CSPDarknet backbone for YOLOv5',
        'pretrained': True,
        'feature_optimization': False,
        'layers': [3, 4, 6, 3],
        'channels': [256, 512, 1024, 2048],
        'input_size': 640
    },
    BackboneType.EFFICIENTNET_B4.value: {
        'name': 'EfficientNet-B4',
        'description': 'EfficientNet-B4 backbone for enhanced feature extraction',
        'pretrained': True,
        'feature_optimization': True,
        'layers': [2, 4, 4, 6, 6, 8, 2],
        'channels': [24, 32, 56, 112, 160, 272, 448],
        'input_size': 640
    }
}

# UI Constants
BACKBONE_UI_TITLE = "🏗️ Model Backbone Configuration"
BACKBONE_UI_SUBTITLE = "Configure and validate model backbone architecture"

# Operation Types
class BackboneOperation(Enum):
    """Available backbone operations"""
    VALIDATE = "validate"
    LOAD = "load"
    BUILD = "build"
    SUMMARY = "summary"

# Progress Tracking Constants
PROGRESS_STEPS = {
    BackboneOperation.VALIDATE.value: [
        "🔍 Validating configuration",
        "📋 Checking backbone compatibility", 
        "✅ Validation complete"
    ],
    BackboneOperation.LOAD.value: [
        "🔄 Loading backbone model",
        "⚙️ Configuring parameters",
        "🎯 Setting up features",
        "✅ Backbone loaded"
    ],
    BackboneOperation.BUILD.value: [
        "🏗️ Building backbone architecture",
        "🔧 Configuring layers",
        "📊 Calculating parameters",
        "✅ Build complete"
    ],
    BackboneOperation.SUMMARY.value: [
        "📊 Generating model summary",
        "📈 Analyzing parameters",
        "✅ Summary complete"
    ]
}

# Error Messages
ERROR_MESSAGES = {
    'invalid_backbone': "Invalid backbone type selected",
    'loading_failed': "Failed to load backbone model",
    'build_failed': "Failed to build backbone architecture",
    'validation_failed': "Backbone configuration validation failed",
    'backend_error': "Backend service error occurred"
}

# Success Messages
SUCCESS_MESSAGES = {
    'validation_success': "Backbone configuration validated successfully",
    'load_success': "Backbone model loaded successfully",
    'build_success': "Backbone architecture built successfully",
    'summary_success': "Model summary generated successfully"
}

# Configuration Keys
CONFIG_KEYS = {
    'backbone_type': 'backbone_type',
    'pretrained': 'pretrained',
    'feature_optimization': 'feature_optimization',
    'custom_config': 'custom_config',
    'advanced_settings': 'advanced_settings'
}

# Button Configuration
BUTTON_CONFIG = {
    'validate': {
        'text': '🔍 Validate',
        'style': 'info',
        'tooltip': 'Validate backbone configuration',
        'order': 1
    },
    'load': {
        'text': '📥 Load Model',
        'style': 'primary',
        'tooltip': 'Load backbone model with current configuration',
        'order': 2
    },
    'build': {
        'text': '🏗️ Build',
        'style': 'success',
        'tooltip': 'Build backbone architecture',
        'order': 3
    },
    'summary': {
        'text': '📊 Summary',
        'style': 'warning',
        'tooltip': 'Generate model summary and statistics',
        'order': 4
    }
}

# Log Levels
LOG_LEVELS = {
    'debug': 'DEBUG',
    'info': 'INFO', 
    'warning': 'WARNING',
    'error': 'ERROR',
    'success': 'SUCCESS'
}

# File Extensions
SUPPORTED_MODEL_EXTENSIONS = ['.pth', '.pt', '.ckpt', '.bin']

# Model Size Limits (in MB)
MODEL_SIZE_LIMITS = {
    'min': 1,
    'max': 2000,
    'recommended': 500
}