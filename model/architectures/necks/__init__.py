"""
File: smartcash/model/architectures/necks/__init__.py
Deskripsi: Package initialization untuk necks
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