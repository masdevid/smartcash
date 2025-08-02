#!/usr/bin/env python3
"""
YOLOv5-based mAP calculator for SmartCash validation phase.

This is a refactored version that follows Single Responsibility Principle
and maintains backward compatibility with the original API. The monolithic
calculator has been split into focused, reusable modules.

MIGRATION NOTE: This file now delegates to the modular implementation.
The original 1000+ line file has been decomposed into:
- YOLOv5UtilitiesManager: Handles YOLOv5 imports and lazy loading
- HierarchicalProcessor: Manages multi-layer confidence modulation  
- MemoryOptimizedProcessor: Platform-aware memory management
- BatchProcessor: Handles batch-level prediction processing
- YOLOv5MapCalculator: Core mAP calculation logic

Uses the built-in YOLOv5 metrics utilities to compute mAP@0.5 during validation.
This provides accurate and standardized object detection metrics.
"""

# Import the refactored implementation
from .yolov5_map_calculator_refactored import (
    YOLOv5MapCalculator,
    create_yolov5_map_calculator,
    DEBUG_HIERARCHICAL
)

# Maintain backward compatibility by re-exporting the legacy accessor functions
from .yolo_utils_manager import (
    get_ap_per_class,
    get_box_iou, 
    get_xywh2xyxy,
    get_non_max_suppression
)

# Re-export all public symbols to maintain API compatibility
__all__ = [
    'YOLOv5MapCalculator',
    'create_yolov5_map_calculator',
    'DEBUG_HIERARCHICAL',
    'get_ap_per_class',
    'get_box_iou',
    'get_xywh2xyxy', 
    'get_non_max_suppression'
]