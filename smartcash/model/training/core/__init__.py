"""
Core training components for the unified training pipeline.

This package contains the refactored training components following
Single Responsibility Principle (SRP) design.
"""

from .training_executor import TrainingExecutor
from .validation.validation_executor import ValidationExecutor
from .prediction.prediction_processor import PredictionProcessor
from .progress_manager import ProgressManager

# Import mAP calculator and utilities
from .yolov5_map_calculator import (
    YOLOv5MapCalculator,  # Now uses Ultralytics backend
    UltralyticsMapCalculator,
    get_box_iou,
    get_xywh2xyxy,
    get_non_max_suppression,
    get_ap_per_class
)

__all__ = [
    # Core components
    'TrainingExecutor', 
    'ValidationExecutor',
    'PredictionProcessor',
    'ProgressManager',
    
    # mAP calculator (now with Ultralytics backend)
    'YOLOv5MapCalculator',
    'UltralyticsMapCalculator',
    
    # Utility functions
    'get_box_iou',
    'get_xywh2xyxy',
    'get_non_max_suppression',
    'get_ap_per_class'
]