"""
File: smartcash/model/config/model_constants.py
Deskripsi: Essential model constants - cleaned up version
Note: Obsolete constants have been removed. Legacy OPTIMIZED_MODELS, DEFAULT_MODEL_CONFIG* 
      have been superseded by YAML-based configuration system.
"""

from typing import Dict, Any, List

# Layer definitions for multi-layer detection system (matches MODEL_ARC.md)
DETECTION_LAYERS = ['layer_1', 'layer_2', 'layer_3']

# Legacy layer mapping for backward compatibility
LEGACY_LAYER_MAPPING = {
    'banknote': 'layer_1',  # Full banknote detection
    'nominal': 'layer_2',   # Denomination-specific features
    'security': 'layer_3'   # Common security features
}

# Detection thresholds for each layer
DETECTION_THRESHOLDS = {
    'layer_1': 0.25,  # Full banknote detection
    'layer_2': 0.30,  # Denomination features
    'layer_3': 0.35   # Security features
}

# Complete layer configuration matching MODEL_ARC.md specification
LAYER_CONFIG = {
    'layer_1': {
        'num_classes': 7,
        'classes': [
            {'id': 0, 'name': '001', 'desc': '1K IDR'},
            {'id': 1, 'name': '002', 'desc': '2K IDR'},
            {'id': 2, 'name': '005', 'desc': '5K IDR'},
            {'id': 3, 'name': '010', 'desc': '10K IDR'},
            {'id': 4, 'name': '020', 'desc': '20K IDR'},
            {'id': 5, 'name': '050', 'desc': '50K IDR'},
            {'id': 6, 'name': '100', 'desc': '100K IDR'},
        ],
        'description': 'Layer 1: Full Banknote Detection - Detects full note bounding boxes'
    },
    'layer_2': {
        'num_classes': 7,
        'classes': [
            {'id': 7, 'name': 'l2_001', 'desc': '1K Features'},
            {'id': 8, 'name': 'l2_002', 'desc': '2K Features'},
            {'id': 9, 'name': 'l2_005', 'desc': '5K Features'},
            {'id': 10, 'name': 'l2_010', 'desc': '10K Features'},
            {'id': 11, 'name': 'l2_020', 'desc': '20K Features'},
            {'id': 12, 'name': 'l2_050', 'desc': '50K Features'},
            {'id': 13, 'name': 'l2_100', 'desc': '100K Features'},
        ],
        'description': 'Layer 2: Denomination Features - Detects denomination-specific visual markers'
    },
    'layer_3': {
        'num_classes': 3,
        'classes': [
            {'id': 14, 'name': 'l3_sign', 'desc': 'BI Logo'},
            {'id': 15, 'name': 'l3_text', 'desc': 'Serial Number & Micro Text'},
            {'id': 16, 'name': 'l3_thread', 'desc': 'Security Thread'},
        ],
        'description': 'Layer 3: Common Features - Detects common features across all notes'
    }
}

# Flat representation of layer configuration
LAYER_CONFIG_FLAT = [
    {'id': 0, 'layer': 'layer_1', 'name': '001', 'desc': '1K IDR'},
    {'id': 1, 'layer': 'layer_1', 'name': '002', 'desc': '2K IDR'},
    {'id': 2, 'layer': 'layer_1', 'name': '005', 'desc': '5K IDR'},
    {'id': 3, 'layer': 'layer_1', 'name': '010', 'desc': '10K IDR'},
    {'id': 4, 'layer': 'layer_1', 'name': '020', 'desc': '20K IDR'},
    {'id': 5, 'layer': 'layer_1', 'name': '050', 'desc': '50K IDR'},
    {'id': 6, 'layer': 'layer_1', 'name': '100', 'desc': '100K IDR'},
    
    {'id': 7, 'layer': 'layer_2', 'name': 'l2_001', 'desc': '1K Features'},
    {'id': 8, 'layer': 'layer_2', 'name': 'l2_002', 'desc': '2K Features'},
    {'id': 9, 'layer': 'layer_2', 'name': 'l2_005', 'desc': '5K Features'},
    {'id': 10, 'layer': 'layer_2', 'name': 'l2_010', 'desc': '10K Features'},
    {'id': 11, 'layer': 'layer_2', 'name': 'l2_020', 'desc': '20K Features'},
    {'id': 12, 'layer': 'layer_2', 'name': 'l2_050', 'desc': '50K Features'},
    {'id': 13, 'layer': 'layer_2', 'name': 'l2_100', 'desc': '100K Features'},
    
    {'id': 14, 'layer': 'layer_3', 'name': 'l3_sign', 'desc': 'BI Logo'},
    {'id': 15, 'layer': 'layer_3', 'name': 'l3_text', 'desc': 'Serial Number & Micro Text'},
    {'id': 16, 'layer': 'layer_3', 'name': 'l3_thread', 'desc': 'Security Thread'}
]

# Legacy backbone constants (still used by some components)
SUPPORTED_EFFICIENTNET_MODELS = ['efficientnet_b4']

EFFICIENTNET_CHANNELS = {
    'efficientnet_b4': [56, 160, 448],
}

# YOLO standard channels for feature maps (P3, P4, P5 stages)
YOLO_CHANNELS = [128, 256, 512]

# Default feature indices for EfficientNet (P3, P4, P5 stages)
DEFAULT_EFFICIENTNET_INDICES = [2, 3, 4]

# YOLOv5 configuration for CSPDarknet backbone (still used by legacy components)
YOLOV5_CONFIG = {
    'yolov5s': {
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
        'feature_indices': [4, 6, 9],  # P3, P4, P5 layers
        'expected_channels': [128, 256, 512],
        'expected_shapes': [(80, 80), (40, 40), (20, 20)],  # for input 640x640
    }
}

# Utility function to get layer information by name
def get_layer_info(layer_name: str) -> Dict[str, Any]:
    """Get layer configuration information by layer name."""
    return LAYER_CONFIG.get(layer_name, {})

# Utility function to get all class names for a layer
def get_layer_classes(layer_name: str) -> List[str]:
    """Get list of class names for a specific layer."""
    layer_info = LAYER_CONFIG.get(layer_name, {})
    return [cls['name'] for cls in layer_info.get('classes', [])]

# Utility function to get total number of classes across all layers
def get_total_classes() -> int:
    """Get total number of classes across all detection layers."""
    return sum(layer['num_classes'] for layer in LAYER_CONFIG.values())