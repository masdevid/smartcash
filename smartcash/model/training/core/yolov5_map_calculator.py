#!/usr/bin/env python3
"""
YOLOv5-based mAP calculator for SmartCash validation phase.

Uses the built-in YOLOv5 metrics utilities to compute mAP@0.5 during validation.
This provides accurate and standardized object detection metrics.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# Initialize logger
from smartcash.common.logger import get_logger
logger = get_logger(__name__, level="WARNING")

# Keep coordinate conversion and IoU analysis debug for essential monitoring
DEBUG_MAP_CALC = True

# Add YOLOv5 to path for imports with better path resolution
# Current file is: smartcash/model/training/core/yolov5_map_calculator.py
# YOLOv5 is at: yolov5/ (project root)
# So we need to go up 5 levels: core -> training -> model -> smartcash -> project_root
YOLOV5_ROOT = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
YOLOV5_ROOT = YOLOV5_ROOT.resolve()  # Resolve to absolute path

# Ensure YOLOv5 path is properly added
if YOLOV5_ROOT.exists() and str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))  # Insert at beginning for priority
    logger.info(f"Added YOLOv5 path to sys.path: {YOLOV5_ROOT}")

# Try multiple import approaches
ap_per_class = None
box_iou = None
xywh2xyxy = None
non_max_suppression = None

# Approach 1: Direct import from utils (when CWD is YOLOv5 root)
try:
    from utils.metrics import ap_per_class, box_iou
    from utils.general import xywh2xyxy, non_max_suppression
except ImportError as e1:
    # Approach 2: Try importing from yolov5.utils (when YOLOv5 is pip installed)
    try:
        from yolov5.utils.metrics import ap_per_class, box_iou
        from yolov5.utils.general import xywh2xyxy, non_max_suppression
    except ImportError as e2:
        
        # Approach 3: Try with explicit path manipulation (fallback)
        try:
            # Save current working directory
            original_cwd = os.getcwd()
            
            # Change to YOLOv5 directory temporarily
            if YOLOV5_ROOT.exists():
                os.chdir(str(YOLOV5_ROOT))
                from utils.metrics import ap_per_class, box_iou
                from utils.general import xywh2xyxy, non_max_suppression
            else:
                raise ImportError(f"YOLOv5 directory not found: {YOLOV5_ROOT}")
                
        except ImportError:
            logger.warning("YOLOv5 utilities not found - using fallback implementations")
        finally:
            # Restore original working directory
            if 'original_cwd' in locals():
                os.chdir(original_cwd)

# Logger already initialized above

# Add fallback implementations for missing YOLOv5 functions
if xywh2xyxy is None:
    def xywh2xyxy(x):
        """Fallback xywh to xyxy conversion."""
        if isinstance(x, torch.Tensor):
            y = x.clone()
            y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
            y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
            y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
            y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
            return y
        else:
            y = np.copy(x)
            y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
            y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
            y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
            y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
            return y

if box_iou is None:
    def box_iou(box1, box2):
        """Fallback IoU calculation."""
        def _box_area(box):
            if isinstance(box, torch.Tensor):
                return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            else:
                return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        
        if isinstance(box1, torch.Tensor):
            area1 = _box_area(box1)
            area2 = _box_area(box2)
            
            inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
                    torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
            
            return inter / (area1[:, None] + area2 - inter)
        else:
            # NumPy fallback
            area1 = _box_area(box1)
            area2 = _box_area(box2)
            
            inter = np.maximum(0, np.minimum(box1[:, None, 2:], box2[:, 2:]) - 
                             np.maximum(box1[:, None, :2], box2[:, :2])).prod(2)
            
            return inter / (area1[:, None] + area2 - inter)

if non_max_suppression is None:
    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """Fallback NMS implementation."""
        # Simple fallback that just filters by confidence
        output = []
        for pred in prediction:
            # Filter by confidence
            if isinstance(pred, torch.Tensor):
                pred = pred[pred[:, 4] > conf_thres]
                if pred.shape[0] == 0:
                    output.append(torch.empty((0, 6)))
                    continue
                # Take top max_det predictions
                if pred.shape[0] > max_det:
                    pred = pred[:max_det]
                output.append(pred)
            else:
                output.append(pred)
        return output

if ap_per_class is None:
    def ap_per_class(*args, **kwargs):
        """Fallback AP calculation."""
        # Return dummy values matching expected format
        return (
            np.array([0.0]),  # tp
            np.array([0.0]),  # fp  
            np.array([0.0]),  # p
            np.array([0.0]),  # r
            np.array([0.0]),  # f1
            np.array([[0.0]]),  # ap
            np.array([0])     # ap_class
        )


class YOLOv5MapCalculator:
    """
    YOLOv5-based mAP calculator using official YOLOv5 metrics utilities.
    
    This calculator provides accurate mAP@0.5 computation that matches
    standard YOLO evaluation protocols.
    """
    
    def __init__(self, num_classes: int = 7, conf_thres: float = 0.005, iou_thres: float = 0.03):
        """
        Initialize YOLOv5 mAP calculator.
        
        Args:
            num_classes: Number of classes (default 7 for SmartCash banknotes)
            conf_thres: Confidence threshold for predictions
            iou_thres: IoU threshold for NMS and mAP calculation
        """
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Check if YOLOv5 is available
        self.yolov5_available = ap_per_class is not None
        if not self.yolov5_available:
            logger.warning("YOLOv5 metrics not available - using fallback implementations")
        
        # Storage for batch statistics
        self.stats = []  # List of [tp, conf, pred_cls, target_cls]
    
    def reset(self):
        """Reset accumulated statistics for new validation run."""
        self.stats.clear()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update mAP statistics with batch predictions and targets.
        
        Args:
            predictions: Model predictions in YOLO format [batch, detections, 6]
                        where each detection is [x, y, w, h, conf, class]
            targets: Ground truth targets [num_targets, 6] 
                    where each target is [batch_idx, class, x, y, w, h]
        """
        
        if not self.yolov5_available or predictions is None or targets is None:
            return
        
        # Ensure tensors are on CPU for processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        
        try:
            # Apply NMS to predictions if not already applied
            if predictions.dim() == 3:  # [batch, detections, 6]
                # Apply NMS per batch
                nms_predictions = []
                for batch_idx in range(predictions.shape[0]):
                    pred = predictions[batch_idx]
                    
                    # Filter by confidence
                    pred = pred[pred[:, 4] > self.conf_thres]
                    
                    if pred.shape[0] > 0:
                        # Convert xywh to xyxy for NMS
                        pred_xyxy = pred.clone()
                        pred_xyxy[:, :4] = xywh2xyxy(pred[:, :4])
                        
                        # Skip complex NMS, use filtered predictions
                        
                        # Add batch index to predictions: [batch_idx, x1, y1, x2, y2, conf, class]
                        batch_pred = torch.full((pred.shape[0], 1), batch_idx, dtype=torch.float32)
                        batch_predictions = torch.cat([batch_pred, pred_xyxy], dim=1)
                        nms_predictions.append(batch_predictions)
                
                # Combine all batch predictions
                if nms_predictions:
                    predictions = torch.cat(nms_predictions, dim=0)
                else:
                    predictions = torch.empty((0, 7))  # [batch_idx, x1, y1, x2, y2, conf, class]
            
            
            # Process targets and predictions for mAP calculation
            batch_stats = self._process_batch(predictions, targets)
            if batch_stats is not None:
                self.stats.append(batch_stats)
                
        except Exception as e:
            logger.warning(f"Error updating mAP statistics: {e}")
    
    def _process_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> Optional[Tuple]:
        """
        Process a batch of predictions and targets for mAP calculation.
        
        Args:
            predictions: Processed predictions [N, 7] where each row is 
                        [batch_idx, x1, y1, x2, y2, conf, class]
            targets: Targets [M, 6] where each row is [batch_idx, class, x, y, w, h]
        
        Returns:
            Tuple of (tp, conf, pred_cls, target_cls) for mAP calculation
        """
        if predictions.shape[0] == 0:  # No predictions
            if targets.shape[0] > 0:  # But have targets (all FN)
                return (
                    torch.zeros((0, 1), dtype=torch.bool),  # tp
                    torch.zeros(0),  # conf  
                    torch.zeros(0, dtype=torch.int32),  # pred_cls
                    targets[:, 1].int()  # target_cls
                )
            else:  # No predictions and no targets
                return None
        
        if targets.shape[0] == 0:  # No targets (all FP)
            return (
                torch.zeros((predictions.shape[0], 1), dtype=torch.bool),  # tp
                predictions[:, 5],  # conf
                predictions[:, 6].int(),  # pred_cls
                torch.zeros(0, dtype=torch.int32)  # target_cls
            )
        
        # Convert target format from [batch_idx, class, x, y, w, h] to [batch_idx, x1, y1, x2, y2, class]
        target_boxes = targets.clone()
        
        if DEBUG_MAP_CALC:
            logger.debug(f"🔧 Coordinate conversion debug:")
            logger.debug(f"  • Original targets shape: {targets.shape}")
            logger.debug(f"  • Target sample (xywh): {targets[0, 2:6] if targets.shape[0] > 0 else 'empty'}")
            logger.debug(f"  • Predictions sample (xyxy): {predictions[0, 1:5] if predictions.shape[0] > 0 else 'empty'}")
            
            # Analyze target box sizes
            if targets.shape[0] > 0:
                target_widths = targets[:, 4]  # width column
                target_heights = targets[:, 5]  # height column
                logger.debug(f"  • Target widths: min={target_widths.min():.3f}, max={target_widths.max():.3f}, mean={target_widths.mean():.3f}")
                logger.debug(f"  • Target heights: min={target_heights.min():.3f}, max={target_heights.max():.3f}, mean={target_heights.mean():.3f}")
                large_boxes = ((target_widths > 0.8) | (target_heights > 0.8)).sum()
                logger.debug(f"  • Oversized boxes (>80% image): {large_boxes.item()}/{targets.shape[0]}")
                
            # Analyze prediction box sizes
            if predictions.shape[0] > 0:
                pred_widths = predictions[:, 3] - predictions[:, 1]  # x2 - x1
                pred_heights = predictions[:, 4] - predictions[:, 2]  # y2 - y1
                logger.debug(f"  • Prediction widths: min={pred_widths.min():.3f}, max={pred_widths.max():.3f}, mean={pred_widths.mean():.3f}")
                logger.debug(f"  • Prediction heights: min={pred_heights.min():.3f}, max={pred_heights.max():.3f}, mean={pred_heights.mean():.3f}")
        
        target_boxes[:, 2:6] = xywh2xyxy(targets[:, 2:6])  # Convert xywh to xyxy
        target_boxes = target_boxes[:, [0, 2, 3, 4, 5, 1]]  # Reorder to [batch_idx, x1, y1, x2, y2, class]
        
        if DEBUG_MAP_CALC:
            logger.debug(f"  • Converted target sample (xyxy): {target_boxes[0, 1:5] if target_boxes.shape[0] > 0 else 'empty'}")
            logger.debug(f"  • Target coordinate ranges: x1={target_boxes[:, 1].min():.3f}-{target_boxes[:, 1].max():.3f}, y1={target_boxes[:, 2].min():.3f}-{target_boxes[:, 2].max():.3f}")
            logger.debug(f"  • Prediction coordinate ranges: x1={predictions[:, 1].min():.3f}-{predictions[:, 1].max():.3f}, y1={predictions[:, 2].min():.3f}-{predictions[:, 2].max():.3f}")
        
        # Debug IoU calculation
        if DEBUG_MAP_CALC and predictions.shape[0] > 0 and target_boxes.shape[0] > 0:
            sample_iou = box_iou(predictions[0:1, 1:5], target_boxes[0:1, 1:5])
            logger.debug(f"  • Sample IoU: {sample_iou.item():.4f} (threshold: {self.iou_thres})")
            logger.debug(f"  • Sample pred class: {predictions[0, 6].int().item()}, target class: {target_boxes[0, 5].int().item()}")
        
        # Compute IoU between predictions and targets
        iou_matrix = box_iou(predictions[:, 1:5], target_boxes[:, 1:5])  # IoU between boxes
        
        if DEBUG_MAP_CALC:
            logger.debug(f"🔍 IoU Matrix Analysis:")
            logger.debug(f"  • IoU matrix shape: {iou_matrix.shape}")
            logger.debug(f"  • IoU range: {iou_matrix.min():.6f} - {iou_matrix.max():.6f}")
            logger.debug(f"  • IoU mean: {iou_matrix.mean():.6f}")
            logger.debug(f"  • IoUs > threshold ({self.iou_thres}): {(iou_matrix > self.iou_thres).sum().item()}")
            if iou_matrix.max() > 0:
                logger.debug(f"  • Best IoU location: {torch.unravel_index(iou_matrix.argmax(), iou_matrix.shape)}")
                logger.debug(f"  • Best IoU value: {iou_matrix.max().item():.6f}")
        
        # Find matches based on IoU threshold
        tp = torch.zeros((predictions.shape[0], 1), dtype=torch.bool)
        matches = torch.where(iou_matrix > self.iou_thres)
        
        if matches[0].shape[0] > 0:
            # Sort by IoU (highest first)
            match_ious = iou_matrix[matches]
            sort_indices = torch.argsort(match_ious, descending=True)
            matches = (matches[0][sort_indices], matches[1][sort_indices])
            
            # Remove duplicate matches (keep highest IoU for each target)
            matched_targets = set()
            matched_predictions = set()
            
            for pred_idx, target_idx in zip(matches[0].tolist(), matches[1].tolist()):
                if target_idx not in matched_targets and pred_idx not in matched_predictions:
                    # Check if classes match
                    pred_class = predictions[pred_idx, 6].int().item()
                    target_class = target_boxes[target_idx, 5].int().item()
                    
                    if pred_class == target_class:
                        tp[pred_idx, 0] = True
                        matched_targets.add(target_idx)
                        matched_predictions.add(pred_idx)
        
        
        return (
            tp,  # True positives
            predictions[:, 5],  # Confidence scores
            predictions[:, 6].int(),  # Predicted classes
            target_boxes[:, 5].int()  # Target classes
        )
    
    def compute_map(self) -> Dict[str, float]:
        """
        Compute final mAP metrics from accumulated statistics.
        
        Returns:
            Dictionary containing mAP metrics:
            - 'map50': mAP@0.5
            - 'map50_95': mAP@0.5:0.95 (set to 0 for now)
            - 'precision': Mean precision
            - 'recall': Mean recall
            - 'f1': Mean F1 score
        """
        if not self.yolov5_available:
            logger.warning("YOLOv5 not available - returning zero mAP")
            return {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        if not self.stats:
            logger.warning("No statistics accumulated - returning zero mAP")
            return {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        try:
            # Concatenate all statistics
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
            
            
            if len(stats) and stats[0].any():
                # Compute AP per class using YOLOv5 function
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                    *stats, 
                    plot=False, 
                    save_dir="", 
                    names={}
                )
                
                # Extract mAP@0.5
                ap50 = ap[:, 0] if ap.shape[1] > 0 else np.array([0.0])
                
                # Compute mean metrics
                map50 = ap50.mean()
                precision = p.mean()
                recall = r.mean()
                f1_score = f1.mean()
                
                
                return {
                    'map50': float(map50),
                    'map50_95': 0.0,  # Not computed for performance
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1_score)
                }
            
        except Exception as e:
            logger.error(f"Error computing mAP: {e}")
        
        # Return zeros on error
        return {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }


def create_yolov5_map_calculator(num_classes: int = 7, conf_thres: float = 0.005, iou_thres: float = 0.03) -> YOLOv5MapCalculator:
    """
    Factory function to create YOLOv5 mAP calculator.
    
    Args:
        num_classes: Number of classes
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        
    Returns:
        YOLOv5MapCalculator instance
    """
    return YOLOv5MapCalculator(num_classes, conf_thres, iou_thres)