"""
YOLOv5 Integration Package
Modular YOLOv5 integration components following SRP principles
"""

from .integration_manager import (
    SmartCashYOLOv5Integration,
    get_integration_manager,
    create_smartcash_yolov5_model,
    create_training_model
)
from .training_compatibility import SmartCashTrainingCompatibilityWrapper
from .memory_manager import YOLOv5MemoryManager
from .config_manager import YOLOv5ConfigManager
from .pretrained_weights import YOLOv5PretrainedWeights
from .model_factory import YOLOv5ModelFactory

# Export key functions and classes
__all__ = [
    'SmartCashYOLOv5Integration',
    'SmartCashTrainingCompatibilityWrapper',
    'create_smartcash_yolov5_model',
    'create_training_model',
    'get_integration_manager',
    'YOLOv5MemoryManager',
    'YOLOv5ConfigManager',
    'YOLOv5PretrainedWeights',
    'YOLOv5ModelFactory'
]