"""
File: smartcash/model/architectures/necks/__init__.py
Deskripsi: Inisialisasi dan export neck modules untuk pemrosesan fitur
"""

from smartcash.model.architectures.necks.fpn_pan import (
    FeatureProcessingNeck,
    FeaturePyramidNetwork,
    PathAggregationNetwork
)

__all__ = [
    'FeatureProcessingNeck',
    'FeaturePyramidNetwork',
    'PathAggregationNetwork'
]