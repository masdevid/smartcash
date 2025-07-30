"""
YOLOv5 Integration Manager for SmartCash
Provides unified interface for using SmartCash architectures with YOLOv5

This module now serves as a compatibility layer for the refactored modular architecture.
The actual implementation has been split into specialized modules in the yolov5/ package.
"""

# Import from the new modular structure
from .yolov5 import (
    SmartCashYOLOv5Integration,
    SmartCashTrainingCompatibilityWrapper,
    create_smartcash_yolov5_model,
    create_training_model,
    get_integration_manager
)

# Export key functions and classes for backward compatibility
__all__ = [
    'SmartCashYOLOv5Integration',
    'SmartCashTrainingCompatibilityWrapper',
    'create_smartcash_yolov5_model',
    'create_training_model',
    'get_integration_manager'
]