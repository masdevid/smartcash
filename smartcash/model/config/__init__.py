"""
File: smartcash/model/config/__init__.py
Deskripsi: Model configuration package - cleaned up version
Note: Legacy ModelConfig and BackboneConfig classes have been removed.
      Use YAML-based configuration system instead.
"""

from smartcash.model.config.model_constants import (
    DETECTION_LAYERS,
    DETECTION_THRESHOLDS,
    LAYER_CONFIG,
    LAYER_CONFIG_FLAT,
    LEGACY_LAYER_MAPPING,
    SUPPORTED_EFFICIENTNET_MODELS,
    EFFICIENTNET_CHANNELS,
    YOLO_CHANNELS,
    YOLOV5_CONFIG,
    DEFAULT_EFFICIENTNET_INDICES,
    get_layer_info,
    get_layer_classes,
    get_total_classes
)

# Export essential constants and utilities
__all__ = [
    # Layer Detection Constants
    'DETECTION_LAYERS',         # Layer deteksi yang didukung
    'DETECTION_THRESHOLDS',     # Threshold default untuk setiap layer
    'LAYER_CONFIG',             # Konfigurasi lengkap untuk setiap layer deteksi
    'LAYER_CONFIG_FLAT',        # Flat representation of layer configuration
    'LEGACY_LAYER_MAPPING',     # Legacy layer mapping for backward compatibility
    
    # Legacy Backbone Constants (still used by some components)
    'SUPPORTED_EFFICIENTNET_MODELS', # Model EfficientNet yang didukung
    'EFFICIENTNET_CHANNELS',    # Channel output untuk setiap stage EfficientNet
    'YOLO_CHANNELS',            # Channel standar YOLOv5 untuk feature maps
    'YOLOV5_CONFIG',            # Konfigurasi model YOLOv5 untuk CSPDarknet
    'DEFAULT_EFFICIENTNET_INDICES', # Default feature indices for EfficientNet
    
    # Utility Functions
    'get_layer_info',           # Get layer configuration by name
    'get_layer_classes',        # Get class names for a layer
    'get_total_classes',        # Get total number of classes
]

# Migration Notes:
# - ModelConfig and BackboneConfig classes have been removed
# - OPTIMIZED_MODELS and DEFAULT_MODEL_CONFIG* constants have been removed
# - Use YAML-based configuration system in smartcash/model/architectures/configs/
# - For model building, use YOLOv5Integration with YAML configs