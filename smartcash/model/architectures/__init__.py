"""
File: smartcash/model/architectures/__init__.py
Deskripsi: Inisialisasi dan export komponen arsitektur model
"""

# Re-export dari backbones submodule
from smartcash.model.architectures.backbones import (
    BaseBackbone,
    EfficientNetBackbone,
    CSPDarknet
)

# Re-export dari necks submodule
from smartcash.model.architectures.necks import (
    FeatureProcessingNeck,
    FeaturePyramidNetwork,
    PathAggregationNetwork
)

# Re-export dari heads submodule
from smartcash.model.architectures.heads import DetectionHead

__all__ = [
    # Backbones
    'BaseBackbone',
    'EfficientNetBackbone',
    'CSPDarknet',
    
    # Necks
    'FeatureProcessingNeck',
    'FeaturePyramidNetwork',
    'PathAggregationNetwork',
    
    # Heads
    'DetectionHead'
]