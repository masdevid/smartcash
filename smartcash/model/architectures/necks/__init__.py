"""
File: smartcash/model/architectures/necks/__init__.py
Deskripsi: Package initialization untuk necks
"""

from smartcash.model.architectures.necks.fpn_pan import (
    FPN_PAN,
    FeaturePyramidNetwork,
    PathAggregationNetwork
)

# Alias FPN_PAN sebagai FeatureProcessingNeck untuk backward compatibility
FeatureProcessingNeck = FPN_PAN

__all__ = [
    'FeatureProcessingNeck',
    'FPN_PAN',
    'FeaturePyramidNetwork',
    'PathAggregationNetwork'
]