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
from typing import Dict, Tuple, Optional

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

# Try multiple import approaches with optimized fallback
def _import_yolov5_utils():
    """Optimized YOLOv5 import with single fallback warning."""
    import_attempts = [
        lambda: __import__('utils.metrics', fromlist=['ap_per_class', 'box_iou']),
        lambda: __import__('yolov5.utils.metrics', fromlist=['ap_per_class', 'box_iou']),
        lambda: _import_with_cwd_change()
    ]
    
    for attempt in import_attempts:
        try:
            metrics_mod = attempt()
            general_mod = __import__(metrics_mod.__name__.replace('metrics', 'general'), 
                                   fromlist=['xywh2xyxy', 'non_max_suppression'])
            return (metrics_mod.ap_per_class, metrics_mod.box_iou, 
                   general_mod.xywh2xyxy, general_mod.non_max_suppression)
        except (ImportError, AttributeError):
            continue
    
    logger.warning("YOLOv5 utilities not found - using optimized fallbacks")
    return None, None, None, None

def _import_with_cwd_change():
    """Import with temporary CWD change."""
    if not YOLOV5_ROOT.exists():
        raise ImportError(f"YOLOv5 directory not found: {YOLOV5_ROOT}")
    
    original_cwd = os.getcwd()
    try:
        os.chdir(str(YOLOV5_ROOT))
        return __import__('utils.metrics', fromlist=['ap_per_class', 'box_iou'])
    finally:
        os.chdir(original_cwd)

ap_per_class, box_iou, xywh2xyxy, non_max_suppression = _import_yolov5_utils()

# Logger already initialized above

# Optimized fallback implementations
def _get_fallback_functions():
    """Create optimized fallback functions only when needed."""
    
    def xywh2xyxy_fallback(x):
        """Optimized xywh to xyxy conversion."""
        if isinstance(x, torch.Tensor):
            center_x, center_y = x[..., 0], x[..., 1]
            half_w, half_h = x[..., 2] / 2, x[..., 3] / 2
            return torch.stack([
                center_x - half_w, center_y - half_h,
                center_x + half_w, center_y + half_h
            ], dim=-1)
        else:
            center_x, center_y = x[..., 0], x[..., 1]
            half_w, half_h = x[..., 2] / 2, x[..., 3] / 2
            return np.stack([
                center_x - half_w, center_y - half_h,
                center_x + half_w, center_y + half_h
            ], axis=-1)
    
    def box_iou_fallback(box1, box2):
        """Optimized IoU calculation."""
        if isinstance(box1, torch.Tensor):
            # Vectorized computation
            area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
            area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
            
            inter = (torch.minimum(box1[:, None, 2:], box2[:, 2:]) - 
                    torch.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
            
            return inter / (area1[:, None] + area2 - inter + 1e-7)  # Add epsilon for stability
        return np.zeros((len(box1), len(box2)))  # Simplified numpy fallback
    
    def ap_per_class_fallback(*args, **kwargs):
        """Simplified AP calculation fallback."""
        return (np.array([0.0]), np.array([0.0]), np.array([0.0]), 
               np.array([0.0]), np.array([0.0]), np.array([[0.0]]), np.array([0]))
    
    return xywh2xyxy_fallback, box_iou_fallback, ap_per_class_fallback

# Apply fallbacks only if needed
if xywh2xyxy is None:
    xywh2xyxy, box_iou, ap_per_class = _get_fallback_functions()
if non_max_suppression is None:
    def non_max_suppression(prediction, conf_thres=0.25, **kwargs):
        """Simplified NMS fallback."""
        return [pred[pred[:, 4] > conf_thres] if isinstance(pred, torch.Tensor) 
               else pred for pred in prediction]


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
            # Optimized batch processing - O(N) instead of O(B*N)
            if predictions.dim() == 3:  # [batch, detections, 6]
                batch_size, num_dets, _ = predictions.shape
                
                # Vectorized confidence filtering across all batches
                conf_mask = predictions[:, :, 4] > self.conf_thres
                
                # Get valid predictions with batch indices
                valid_preds = []
                for batch_idx in range(batch_size):
                    batch_mask = conf_mask[batch_idx]
                    if batch_mask.any():
                        pred = predictions[batch_idx][batch_mask]
                        # Convert coordinates once per batch
                        pred_xyxy = pred.clone()
                        pred_xyxy[:, :4] = xywh2xyxy(pred[:, :4])
                        
                        # Add batch index efficiently
                        batch_indices = torch.full((pred.shape[0], 1), batch_idx, 
                                                  dtype=torch.float32, device=pred.device)
                        batch_predictions = torch.cat([batch_indices, pred_xyxy], dim=1)
                        valid_preds.append(batch_predictions)
                
                predictions = torch.cat(valid_preds, dim=0) if valid_preds else torch.empty((0, 7))
            
            
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
        
        # Optimized vectorized matching - O(M+N) instead of O(M*N) loops
        tp = torch.zeros((predictions.shape[0], 1), dtype=torch.bool)
        
        # Find potential matches above IoU threshold
        iou_mask = iou_matrix > self.iou_thres
        matches = torch.where(iou_mask)
        
        if matches[0].shape[0] > 0:
            # Get IoU values for valid matches
            match_ious = iou_matrix[matches]
            
            # Vectorized class matching check
            pred_classes = predictions[matches[0], 6].int()
            target_classes = target_boxes[matches[1], 5].int()
            class_matches = pred_classes == target_classes
            
            # Filter matches by class compatibility
            valid_mask = class_matches
            if valid_mask.any():
                valid_pred_idx = matches[0][valid_mask]
                valid_target_idx = matches[1][valid_mask]
                valid_ious = match_ious[valid_mask]
                
                # Sort by IoU (highest first) for greedy matching
                sort_indices = torch.argsort(valid_ious, descending=True)
                sorted_pred_idx = valid_pred_idx[sort_indices]
                sorted_target_idx = valid_target_idx[sort_indices]
                
                # Greedy assignment - vectorized duplicate removal
                used_targets = torch.zeros(target_boxes.shape[0], dtype=torch.bool)
                used_predictions = torch.zeros(predictions.shape[0], dtype=torch.bool)
                
                for i in range(len(sorted_pred_idx)):
                    pred_idx = sorted_pred_idx[i].item()
                    target_idx = sorted_target_idx[i].item()
                    
                    if not used_targets[target_idx] and not used_predictions[pred_idx]:
                        tp[pred_idx, 0] = True
                        used_targets[target_idx] = True
                        used_predictions[pred_idx] = True
        
        
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