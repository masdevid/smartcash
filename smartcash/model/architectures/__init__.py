"""
SmartCash Model Architectures
YOLOv5-Compatible Multi-Layer Detection System for Banknote Detection
"""

# YOLOv5 Integration Components
from smartcash.model.architectures.yolov5_integration import (
    SmartCashYOLOv5Integration,
    SmartCashTrainingCompatibilityWrapper,
    create_smartcash_yolov5_model,
    create_training_model,
    get_integration_manager
)

# Backbone Components
from smartcash.model.architectures.backbones import (
    BaseBackbone,
    EfficientNetBackbone,
    CSPDarknet,
)

# YOLOv5-Compatible Backbone Adapters
from smartcash.model.architectures.backbones.yolov5_backbone import (
    YOLOv5BackboneAdapter,
    YOLOv5CSPDarknetAdapter,
    YOLOv5EfficientNetAdapter,
    YOLOv5BackboneFactory
)

# Neck Components  
from smartcash.model.architectures.necks import (
    FeatureProcessingNeck,
    FeaturePyramidNetwork,
    PathAggregationNetwork,
)

# YOLOv5-Compatible Neck
from smartcash.model.architectures.necks.yolov5_neck import YOLOv5FPNPANNeck

# Detection Heads
from smartcash.model.architectures.heads import (
    YOLOv5MultiLayerDetect,
    YOLOv5HeadAdapter,
    MultiLayerHead,
    ChannelAttention
)

__all__ = [
    # YOLOv5 Integration
    'SmartCashYOLOv5Integration',
    'SmartCashTrainingCompatibilityWrapper', 
    'create_smartcash_yolov5_model',
    'create_training_model',
    'get_integration_manager',
    
    # Backbones
    'BaseBackbone',
    'EfficientNetBackbone', 
    'CSPDarknet',
    'YOLOv5BackboneAdapter',
    'YOLOv5CSPDarknetAdapter',
    'YOLOv5EfficientNetAdapter',
    'YOLOv5BackboneFactory',
    
    # Necks
    'FeatureProcessingNeck',
    'FeaturePyramidNetwork',
    'PathAggregationNetwork',
    'YOLOv5FPNPANNeck',
    
    # Heads
    'YOLOv5MultiLayerDetect',
    'YOLOv5HeadAdapter',
    'MultiLayerHead',
    'ChannelAttention',
]