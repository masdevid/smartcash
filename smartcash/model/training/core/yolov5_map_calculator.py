#!/usr/bin/env python3
"""
Ultralytics-based mAP calculator for SmartCash validation.

This module provides a modern implementation of mAP calculation using the
Ultralytics package with enhanced features:
- mAP50 and mAP50-95 computation
- Modern NMS implementation
- Progressive confidence/IoU thresholds
- Platform-aware optimizations
- Enhanced debug logging

This is the main entry point for mAP calculation in the SmartCash pipeline.
"""

from typing import Dict, Any, Optional

from .ultralytics_map_calculator import UltralyticsMapCalculator
from .ultralytics_utils_manager import (
    get_box_iou, 
    get_xywh2xyxy,
    get_non_max_suppression,
    get_ap_per_class
)

# Alias for backward compatibility
YOLOv5MapCalculator = UltralyticsMapCalculator


def create_yolov5_map_calculator(
    num_classes: int = 17,
    conf_thres: float = 0.001,
    iou_thres: float = 0.5,
    debug: bool = False,
    training_context: Optional[Dict[str, Any]] = None,
    use_progressive_thresholds: bool = True,
    use_standard_map: bool = True
) -> UltralyticsMapCalculator:
    """
    Create mAP calculator with Ultralytics backend.
    
    Args:
        num_classes: Number of classes (default: 17 for SmartCash)
        conf_thres: Base confidence threshold (0.001-1.0)
        iou_thres: Base IoU threshold (0.1-0.95)
        debug: Enable debug logging
        training_context: Optional training context for metrics
        use_progressive_thresholds: Enable progressive threshold scheduling
        use_standard_map: Use standard mAP calculation (recommended)
        
    Returns:
        Configured UltralyticsMapCalculator instance
    """
    from .ultralytics_map_calculator import create_ultralytics_map_calculator
    
    return create_ultralytics_map_calculator(
        num_classes=num_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        debug=debug,
        training_context=training_context or {},
        use_progressive_thresholds=use_progressive_thresholds,
        use_standard_map=use_standard_map
    )


# Re-export public API
__all__ = [
    'YOLOv5MapCalculator',
    'UltralyticsMapCalculator',
    'create_yolov5_map_calculator',
    'get_box_iou',
    'get_xywh2xyxy',
    'get_non_max_suppression',
    'get_ap_per_class'
]