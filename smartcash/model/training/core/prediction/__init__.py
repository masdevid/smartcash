"""
Prediction processing components for SmartCash training pipeline.

This package contains unified prediction processing components that eliminate
code duplication and provide consistent target processing across the codebase.
"""

from .prediction_processor import PredictionProcessor
from .classification_extractor import ClassificationExtractor
from .target_processor import TargetProcessor
from .target_format_converter import TargetFormatConverter
from .prediction_cache import PredictionCache

__all__ = [
    'PredictionProcessor',
    'ClassificationExtractor', 
    'TargetProcessor',
    'TargetFormatConverter',
    'PredictionCache'
]