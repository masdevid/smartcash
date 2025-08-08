#!/usr/bin/env python3
"""
Modern mAP calculator using Ultralytics for SmartCash validation.

Replaced YOLOv5 utils with official Ultralytics package for:
- Enhanced mAP calculation with mAP50-95 support
- Modern NMS implementation  
- Improved performance and reliability
- Standardized evaluation metrics
- Better error handling

Maintains backward compatibility with existing YOLOv5MapCalculator API.
"""

# Backward compatibility module - imports handled by submodules

# Import modern Ultralytics implementation
from .ultralytics_map_calculator import (
    UltralyticsMapCalculator,
    create_ultralytics_map_calculator
)

# Re-export utility functions with Ultralytics backend
from .ultralytics_utils_manager import (
    get_box_iou, 
    get_xywh2xyxy,
    get_non_max_suppression,
    get_ap_per_class
)

# Backward compatibility alias - now uses Ultralytics backend
YOLOv5MapCalculator = UltralyticsMapCalculator

# Factory function with backward compatibility
def create_yolov5_map_calculator(
    num_classes: int = 17,
    conf_thres: float = 0.001,
    iou_thres: float = 0.5,
    debug: bool = False,
    training_context: dict = None,
    use_progressive_thresholds: bool = True,
    use_standard_map: bool = True
) -> UltralyticsMapCalculator:
    """
    Create mAP calculator with Ultralytics backend (backward compatible).
    
    Args:
        num_classes: Number of classes
        conf_thres: Base confidence threshold  
        iou_thres: Base IoU threshold
        debug: Enable debug logging
        training_context: Training context information
        use_progressive_thresholds: Enable progressive threshold scheduling
        use_standard_map: Use standard mAP calculation
        
    Returns:
        UltralyticsMapCalculator instance (compatible with YOLOv5MapCalculator API)
    """
    return create_ultralytics_map_calculator(
        num_classes=num_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        debug=debug,
        training_context=training_context,
        use_progressive_thresholds=use_progressive_thresholds,
        use_standard_map=use_standard_map
    )

# Re-export all public symbols to maintain API compatibility
__all__ = [
    'YOLOv5MapCalculator',  # Now uses Ultralytics backend
    'UltralyticsMapCalculator',
    'create_yolov5_map_calculator',
    'create_ultralytics_map_calculator',
    'get_box_iou',
    'get_xywh2xyxy', 
    'get_non_max_suppression',
    'get_ap_per_class'
]