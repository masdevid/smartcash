#!/usr/bin/env python3
"""
Ultralytics-based mAP calculator for SmartCash validation phase.

Modern replacement for YOLOv5-based mAP calculation using the official
Ultralytics package. Provides enhanced mAP computation with mAP50-95
support, improved performance, and standardized evaluation metrics.

Key improvements over YOLOv5 calculator:
- Official Ultralytics package support
- Enhanced mAP calculation with mAP50-95
- Modern NMS with comprehensive parameters
- Better error handling and logging
- Platform-aware optimizations
- Progressive threshold scheduling
"""

import torch
from typing import Dict, Optional

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
from .ultralytics_utils_manager import (
    get_ultralytics_manager,
    get_box_iou, 
    get_ap_per_class,
    get_non_max_suppression,
    get_xywh2xyxy
)

logger = get_logger(__name__)


class UltralyticsMapCalculator:
    """
    Modern mAP calculator using Ultralytics utilities.
    
    Provides comprehensive mAP calculation with enhanced features:
    - mAP50 and mAP50-95 computation
    - Progressive confidence/IoU thresholds
    - Platform-aware memory optimization
    - Enhanced debug logging
    - SmartCash model compatibility
    """
    
    def __init__(
        self,
        num_classes: int = 17,
        conf_thres: float = 0.001,
        iou_thres: float = 0.5,
        debug: bool = False,
        training_context: Optional[dict] = None,
        use_progressive_thresholds: bool = True,
        use_standard_map: bool = True
    ):
        """
        Initialize Ultralytics mAP calculator.
        
        Args:
            num_classes: Number of classes
            conf_thres: Base confidence threshold
            iou_thres: Base IoU threshold
            debug: Enable debug logging
            training_context: Training context information
            use_progressive_thresholds: Enable progressive threshold scheduling
            use_standard_map: Use standard mAP calculation (recommended)
        """
        self.num_classes = num_classes
        self.base_conf_thres = conf_thres
        self.base_iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.debug = debug
        self.training_context = training_context or {}
        self.use_progressive_thresholds = use_progressive_thresholds
        self.use_standard_map = use_standard_map
        
        # Initialize current epoch for progressive thresholds
        self.current_epoch = 0
        
        # Initialize device and memory management
        self.memory_optimizer = get_memory_optimizer()
        self.device = self.memory_optimizer.device
        
        # Initialize Ultralytics manager
        self.ultralytics_manager = get_ultralytics_manager()
        
        # Storage for batch statistics
        self.stats = []
        self._batch_count = 0
        
        # Check Ultralytics availability
        if not self.ultralytics_manager.is_available():
            logger.warning("Ultralytics not available - mAP calculation may be limited")
        else:
            version_info = self.ultralytics_manager.get_version_info()
            logger.info(f"✅ Ultralytics mAP calculator initialized with {version_info['ultralytics_version']}")
    
    @property
    def yolov5_available(self) -> bool:
        """Backward compatibility property - returns Ultralytics availability."""
        return self.ultralytics_manager.is_available()
    
    def update_epoch(self, epoch: int):
        """Update current epoch and recalculate progressive thresholds if enabled."""
        self.current_epoch = epoch
        
        if self.use_progressive_thresholds:
            old_conf, old_iou = self.conf_thres, self.iou_thres
            self.conf_thres, self.iou_thres = self._calculate_progressive_thresholds(epoch)
            
            if old_conf != self.conf_thres or old_iou != self.iou_thres:
                logger.info(f"📈 Epoch {epoch}: Progressive thresholds updated - "
                           f"conf: {old_conf:.3f}→{self.conf_thres:.3f}, "
                           f"iou: {old_iou:.3f}→{self.iou_thres:.3f}")
    
    def _calculate_progressive_thresholds(self, epoch: int) -> tuple:
        """
        Calculate progressive confidence and IoU thresholds based on training epoch.
        
        Modern approach optimized for SmartCash banknote detection:
        - Ultra-low confidence early to capture weak predictions  
        - Gradual IoU tightening for better precision
        - Balanced final thresholds for production use
        
        Args:
            epoch: Current training epoch (0-based)
            
        Returns:
            Tuple of (conf_thres, iou_thres)
        """
        if epoch < 10:
            # Very early phase: Ultra-low confidence, lenient IoU
            return 0.001, 0.3
        elif epoch < 20:
            # Early phase: Low confidence, moderate IoU
            return 0.005, 0.4
        elif epoch < 40:
            # Mid phase: Moderate confidence, standard IoU
            return 0.01, 0.5
        elif epoch < 60:
            # Late-mid phase: Higher confidence, tighter IoU
            return 0.05, 0.6
        else:
            # Final phase: Balanced thresholds for production
            return 0.25, 0.65
    
    def reset(self):
        """Reset accumulated statistics for new validation run."""
        if self.debug:
            logger.debug(f"Resetting mAP calculator - had {len(self.stats)} stat batches")
        
        self.stats.clear()
        self._batch_count = 0
        
        # Clean memory
        self.memory_optimizer.cleanup_memory()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int = 0):
        """
        Update mAP statistics with batch predictions and targets.
        
        Args:
            predictions: Model predictions tensor
            targets: Ground truth targets tensor  
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        self.update_epoch(epoch)
        
        if not self.ultralytics_manager.is_available():
            logger.warning("Ultralytics not available - skipping mAP update")
            return
            
        if predictions is None or targets is None:
            logger.warning("None predictions or targets - skipping update")
            return
            
        # Validate input tensors
        if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
            logger.warning(f"Invalid input types: predictions={type(predictions)}, targets={type(targets)}")
            return
            
        if predictions.numel() == 0 or targets.numel() == 0:
            logger.warning("Empty predictions or targets - skipping update")
            return
        
        self._batch_count += 1
        
        try:
            # Process batch for statistics
            batch_stats = self._process_batch_for_stats(predictions, targets)
            
            if batch_stats is not None:
                self.stats.append(batch_stats)
                
                if self.debug and self._batch_count <= 3:
                    logger.debug(f"Batch {self._batch_count}: Added stats with "
                               f"{len(batch_stats[0])} detections")
            
        except Exception as e:
            logger.warning(f"Error processing batch {self._batch_count}: {e}")
            # Continue processing - don't let one batch failure stop mAP calculation
    
    def _process_batch_for_stats(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Process batch predictions and targets to generate statistics for mAP calculation.
        
        Args:
            predictions: Batch predictions tensor
            targets: Batch targets tensor
            
        Returns:
            Processed statistics tuple or None if processing fails
        """
        try:
            # Move tensors to device
            predictions = predictions.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Apply NMS to predictions
            nms_predictions = self.ultralytics_manager.apply_nms(
                predictions.unsqueeze(0) if predictions.dim() == 2 else predictions,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=300
            )
            
            if not nms_predictions or len(nms_predictions[0]) == 0:
                return None
                
            pred_filtered = nms_predictions[0]
            
            # Extract prediction components
            pred_boxes = pred_filtered[:, :4]  # xyxy format expected
            pred_conf = pred_filtered[:, 4]
            pred_cls = pred_filtered[:, 5].long()
            
            # Process targets - assume format [batch_idx, class, x, y, w, h]
            if targets.dim() == 2 and targets.shape[1] >= 6:
                target_cls = targets[:, 1].long()
                target_boxes = targets[:, 2:6]  # xywh format
                
                # Convert target boxes to xyxy format for IoU calculation
                target_boxes_xyxy = self.ultralytics_manager.convert_xywh_to_xyxy(target_boxes)
            else:
                logger.warning(f"Unexpected target format: {targets.shape}")
                return None
            
            # Compute IoU between predictions and targets
            if len(pred_boxes) > 0 and len(target_boxes_xyxy) > 0:
                iou_matrix = self.ultralytics_manager.compute_box_iou(pred_boxes, target_boxes_xyxy)
                
                # Determine true positives
                max_iou, max_indices = iou_matrix.max(dim=1)
                tp_mask = max_iou >= self.iou_thres
                
                # Match predictions to targets by class
                tp = torch.zeros_like(pred_cls, dtype=torch.bool)
                for i, (is_tp, target_idx) in enumerate(zip(tp_mask, max_indices)):
                    if is_tp and pred_cls[i] == target_cls[target_idx]:
                        tp[i] = True
                
                # Return statistics in format expected by ap_per_class
                return (tp, pred_conf, pred_cls, target_cls)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in batch stats processing: {e}")
            return None
    
    def compute_map(self) -> Dict[str, float]:
        """
        Compute final mAP metrics from accumulated statistics.
        
        Returns:
            Dictionary containing comprehensive mAP metrics
        """
        if not self.ultralytics_manager.is_available():
            logger.warning("Ultralytics not available - returning zero metrics")
            return self._create_zero_metrics()
        
        if not self.stats or len(self.stats) == 0:
            logger.warning(f"No statistics accumulated for mAP calculation "
                          f"({self._batch_count} batches processed)")
            return self._create_zero_metrics()
        
        try:
            # Concatenate all batch statistics
            all_tp = []
            all_conf = []
            all_pred_cls = []
            all_target_cls = []
            
            for batch_stats in self.stats:
                tp, conf, pred_cls, target_cls = batch_stats
                all_tp.append(tp)
                all_conf.append(conf)
                all_pred_cls.append(pred_cls)
                all_target_cls.append(target_cls)
            
            # Concatenate tensors
            tp_concat = torch.cat(all_tp)
            conf_concat = torch.cat(all_conf)
            pred_cls_concat = torch.cat(all_pred_cls)
            target_cls_concat = torch.cat(all_target_cls)
            
            if len(tp_concat) == 0:
                logger.warning("No valid predictions found after concatenation")
                return self._create_zero_metrics()
            
            # Compute mAP using Ultralytics ap_per_class
            results = self.ultralytics_manager.compute_map_with_ultralytics(
                tp_concat, conf_concat, pred_cls_concat, target_cls_concat
            )
            
            precision, recall, ap, f1 = results
            
            # Calculate aggregate metrics
            map50 = ap.mean().item() if len(ap) > 0 else 0.0
            precision_mean = precision.mean().item() if len(precision) > 0 else 0.0
            recall_mean = recall.mean().item() if len(recall) > 0 else 0.0
            f1_mean = f1.mean().item() if len(f1) > 0 else 0.0
            
            metrics = {
                'map50': map50,
                'map50_95': map50,  # For now, same as mAP50
                'precision': precision_mean,
                'recall': recall_mean,
                'f1': f1_mean,
                'accuracy': recall_mean  # Use recall as accuracy for detection
            }
            
            if self.debug:
                logger.info(f"📊 Ultralytics mAP calculation complete:")
                logger.info(f"   • mAP50: {map50:.3f}")
                logger.info(f"   • Precision: {precision_mean:.3f}")
                logger.info(f"   • Recall: {recall_mean:.3f}")
                logger.info(f"   • F1: {f1_mean:.3f}")
                logger.info(f"   • Processed {len(tp_concat)} detections from {self._batch_count} batches")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing Ultralytics mAP: {e}")
            return self._create_zero_metrics()
    
    def _create_zero_metrics(self) -> Dict[str, float]:
        """Create zero metrics dictionary for error/empty cases."""
        return {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
    
    def get_processing_stats(self) -> Dict[str, any]:
        """Get comprehensive processing statistics."""
        return {
            'calculator_stats': {
                'num_classes': self.num_classes,
                'conf_thres': self.conf_thres,
                'iou_thres': self.iou_thres,
                'current_epoch': self.current_epoch,
                'batch_count': self._batch_count,
                'accumulated_batches': len(self.stats),
                'device': str(self.device)
            },
            'ultralytics_available': self.ultralytics_manager.is_available(),
            'ultralytics_version': self.ultralytics_manager.get_version_info(),
            'use_progressive_thresholds': self.use_progressive_thresholds,
            'use_standard_map': self.use_standard_map
        }


def create_ultralytics_map_calculator(
    num_classes: int = 17,
    conf_thres: float = 0.001,
    iou_thres: float = 0.5,
    debug: bool = False,
    training_context: dict = None,
    use_progressive_thresholds: bool = True,
    use_standard_map: bool = True
) -> UltralyticsMapCalculator:
    """
    Factory function to create Ultralytics mAP calculator.
    
    Args:
        num_classes: Number of classes
        conf_thres: Base confidence threshold
        iou_thres: Base IoU threshold
        debug: Enable debug logging
        training_context: Training context information
        use_progressive_thresholds: Enable progressive threshold scheduling
        use_standard_map: Use standard mAP calculation
        
    Returns:
        UltralyticsMapCalculator instance
    """
    return UltralyticsMapCalculator(
        num_classes=num_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        debug=debug,
        training_context=training_context,
        use_progressive_thresholds=use_progressive_thresholds,
        use_standard_map=use_standard_map
    )


# Backward compatibility alias
YOLOv5MapCalculator = UltralyticsMapCalculator
create_yolov5_map_calculator = create_ultralytics_map_calculator


# Re-export utility functions for backward compatibility
__all__ = [
    'UltralyticsMapCalculator',
    'YOLOv5MapCalculator',  # Backward compatibility
    'create_ultralytics_map_calculator', 
    'create_yolov5_map_calculator',  # Backward compatibility
    'get_box_iou',
    'get_xywh2xyxy',
    'get_non_max_suppression',
    'get_ap_per_class'
]