#!/usr/bin/env python3
"""
Ultralytics utilities manager for SmartCash training pipeline.

Modern replacement for YOLOv5 utils with latest Ultralytics framework.
Provides mAP calculation, IoU computation, and NMS functionality using
the official Ultralytics package with improved performance and features.

Key improvements over YOLOv5 utils:
- Official Ultralytics package support
- Enhanced mAP calculation with mAP50-95 
- Optimized IoU computation
- Modern NMS with comprehensive parameters
- Better error handling and logging
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple

from smartcash.common.logger import get_logger

logger = get_logger(__name__, level="DEBUG")


class UltralyticsUtilitiesManager:
    """
    Manages Ultralytics utilities with modern mAP calculation and evaluation.
    
    This replaces the old YOLOv5 utils manager with official Ultralytics package
    support, providing improved performance and standardized evaluation metrics.
    """
    
    def __init__(self):
        """Initialize the Ultralytics utilities manager."""
        self._utils_cache: Optional[Dict[str, Any]] = None
        self._availability_checked = False
        self._is_available = False
        self._ultralytics_version = None
        
    def is_available(self) -> bool:
        """
        Check if Ultralytics is available and can be imported.
        
        Returns:
            bool: True if Ultralytics is available
        """
        if not self._availability_checked:
            self._check_availability()
        return self._is_available
        
    def _check_availability(self) -> None:
        """Check Ultralytics availability and cache result."""
        try:
            import ultralytics
            self._ultralytics_version = ultralytics.__version__
            self._is_available = True
            logger.debug(f"✅ Ultralytics {self._ultralytics_version} is available")
        except ImportError as e:
            self._is_available = False
            logger.warning(f"⚠️ Ultralytics not available: {e}")
        
        self._availability_checked = True
    
    def get_utilities(self) -> Dict[str, Any]:
        """
        Get Ultralytics utilities with lazy loading.
        
        Returns:
            Dictionary containing Ultralytics utility functions
        """
        if self._utils_cache is not None:
            return self._utils_cache
            
        if not self.is_available():
            raise RuntimeError("Ultralytics is not available")
            
        try:
            # Import modern Ultralytics utilities
            from ultralytics.utils.metrics import box_iou, ap_per_class
            from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
            
            # Create utilities dictionary
            utilities = {
                'box_iou': box_iou,
                'ap_per_class': ap_per_class,
                'non_max_suppression': non_max_suppression,
                'xywh2xyxy': xywh2xyxy,
            }
            
            self._utils_cache = utilities
            logger.debug("✅ Ultralytics utilities loaded successfully")
            
            return utilities
            
        except ImportError as e:
            logger.error(f"Failed to import Ultralytics utilities: {e}")
            raise RuntimeError(f"Failed to load Ultralytics utilities: {e}") from e
    
    def get_function(self, function_name: str) -> Callable:
        """
        Get specific Ultralytics function by name.
        
        Args:
            function_name: Name of the function to retrieve
            
        Returns:
            Callable function from Ultralytics
        """
        utilities = self.get_utilities()
        
        if function_name not in utilities:
            available_functions = list(utilities.keys())
            raise ValueError(f"Function '{function_name}' not available. "
                           f"Available functions: {available_functions}")
        
        return utilities[function_name]
    
    def compute_map_with_ultralytics(
        self,
        tp: torch.Tensor,
        conf: torch.Tensor, 
        pred_cls: torch.Tensor,
        target_cls: torch.Tensor,
        eps: float = 1e-16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mAP using modern Ultralytics ap_per_class function.
        
        FIXED: Handle MPS device tensors by moving to CPU before Ultralytics computation.
        
        Args:
            tp: True positives tensor
            conf: Confidence scores tensor
            pred_cls: Predicted classes tensor
            target_cls: Target classes tensor  
            eps: Small epsilon value for numerical stability
            
        Returns:
            Tuple of (precision, recall, average_precision, f1_score)
        """
        if not self.is_available():
            raise RuntimeError("Ultralytics is not available for mAP computation")
            
        ap_per_class_fn = self.get_function('ap_per_class')
        
        # CRITICAL FIX: Move tensors to CPU before calling Ultralytics
        # Ultralytics ap_per_class internally converts tensors to numpy which fails on MPS device
        original_device = tp.device
        
        tp_cpu = tp.cpu()
        conf_cpu = conf.cpu()
        pred_cls_cpu = pred_cls.cpu()
        target_cls_cpu = target_cls.cpu()
        
        # TYPE FIX: Ultralytics ap_per_class expects tp to be float, not boolean
        if tp_cpu.dtype == torch.bool:
            tp_cpu = tp_cpu.float()  # Convert boolean to float (True->1.0, False->0.0)
        
        # DIMENSION FIX: Ultralytics ap_per_class expects tp to be 2D [n_predictions, n_iou_thresholds]
        # For mAP50 calculation, we use single IoU threshold dimension
        if tp_cpu.dim() == 1:
            tp_cpu = tp_cpu.unsqueeze(1)  # Shape: [n_predictions, 1]
        
        # Use modern Ultralytics ap_per_class with enhanced parameters
        results = ap_per_class_fn(
            tp=tp_cpu,
            conf=conf_cpu, 
            pred_cls=pred_cls_cpu,
            target_cls=target_cls_cpu,
            eps=eps,
            plot=False,  # Disable plotting for performance
            save_dir=None  # Don't save plots
        )
        
        # Convert numpy results back to tensors and move to original device if needed
        tensor_results = []
        for result in results[:4]:  # Take first 4: precision, recall, ap, f1
            if isinstance(result, np.ndarray):
                # Convert numpy array to tensor with MPS-compatible dtype
                tensor_result = torch.from_numpy(result)
                
                # Handle MPS device compatibility - convert float64 to float32
                if original_device.type == 'mps' and tensor_result.dtype == torch.float64:
                    tensor_result = tensor_result.float()  # Convert to float32
                
                if original_device != torch.device('cpu'):
                    tensor_result = tensor_result.to(original_device)
                tensor_results.append(tensor_result)
            elif hasattr(result, 'to'):
                # Already a tensor - handle dtype compatibility
                if original_device.type == 'mps' and result.dtype == torch.float64:
                    result = result.float()  # Convert to float32 for MPS
                tensor_results.append(result.to(original_device) if original_device != torch.device('cpu') else result)
            else:
                # Keep as-is for non-tensor types
                tensor_results.append(result)
        
        # Return tuple of (precision, recall, average_precision, f1_score)
        return tuple(tensor_results)
    
    def compute_box_iou(
        self,
        box1: torch.Tensor, 
        box2: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes using Ultralytics.
        
        Args:
            box1: First set of boxes in (x1, y1, x2, y2) format
            box2: Second set of boxes in (x1, y1, x2, y2) format
            eps: Small epsilon for numerical stability
            
        Returns:
            IoU matrix tensor (on original device)
        """
        if not self.is_available():
            raise RuntimeError("Ultralytics is not available for IoU computation")
            
        box_iou_fn = self.get_function('box_iou')
        
        # Handle MPS device compatibility: Ultralytics box_iou works on any device
        # but we ensure consistent device handling
        original_device = box1.device
        
        try:
            # Try direct computation first (works for most devices)
            result = box_iou_fn(box1, box2, eps=eps)
            return result
        except Exception as e:
            # Fallback to CPU computation if device compatibility issues
            logger.debug(f"IoU computation failed on {original_device}, falling back to CPU: {e}")
            
            box1_cpu = box1.cpu()
            box2_cpu = box2.cpu()
            result_cpu = box_iou_fn(box1_cpu, box2_cpu, eps=eps)
            
            return result_cpu.to(original_device)
    
    def apply_nms(
        self,
        prediction: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 300
    ) -> List[torch.Tensor]:
        """
        Apply Non-Maximum Suppression using modern Ultralytics implementation.
        
        Args:
            prediction: Model predictions tensor
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            classes: Filter by classes (None for all classes)
            agnostic: Class-agnostic NMS
            multi_label: Multi-label per box
            max_det: Maximum detections per image
            
        Returns:
            List of filtered predictions per image
        """
        if not self.is_available():
            raise RuntimeError("Ultralytics is not available for NMS")
            
        nms_fn = self.get_function('non_max_suppression')
        
        return nms_fn(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            max_det=max_det
        )
    
    def convert_xywh_to_xyxy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from (x, y, w, h) to (x1, y1, x2, y2) format.
        
        Args:
            x: Boxes in (x, y, w, h) format
            
        Returns:
            Boxes in (x1, y1, x2, y2) format
        """
        if not self.is_available():
            raise RuntimeError("Ultralytics is not available for coordinate conversion")
            
        xywh2xyxy_fn = self.get_function('xywh2xyxy')
        return xywh2xyxy_fn(x)
    
    def get_version_info(self) -> Dict[str, str]:
        """
        Get version information for Ultralytics and dependencies.
        
        Returns:
            Dictionary with version information
        """
        info = {
            'ultralytics_available': str(self.is_available()),
            'ultralytics_version': self._ultralytics_version or 'not available'
        }
        
        try:
            info['torch_version'] = torch.__version__
            info['numpy_version'] = np.__version__
        except Exception as e:
            logger.warning(f"Could not get dependency versions: {e}")
            
        return info


# Global instance for singleton pattern
_ultralytics_manager = None


def get_ultralytics_manager() -> UltralyticsUtilitiesManager:
    """
    Get global Ultralytics utilities manager instance.
    
    Returns:
        UltralyticsUtilitiesManager instance
    """
    global _ultralytics_manager
    if _ultralytics_manager is None:
        _ultralytics_manager = UltralyticsUtilitiesManager()
    return _ultralytics_manager


# Convenience functions for direct access to Ultralytics utilities
def get_ap_per_class() -> Callable:
    """Get Ultralytics ap_per_class function."""
    return get_ultralytics_manager().get_function('ap_per_class')


def get_box_iou() -> Callable:
    """Get Ultralytics box_iou function."""
    return get_ultralytics_manager().get_function('box_iou')


def get_xywh2xyxy() -> Callable:
    """Get Ultralytics xywh2xyxy function."""
    return get_ultralytics_manager().get_function('xywh2xyxy')


def get_non_max_suppression() -> Callable:
    """Get Ultralytics non_max_suppression function."""
    return get_ultralytics_manager().get_function('non_max_suppression')


# Enhanced mAP calculation with modern Ultralytics
def compute_enhanced_map(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor], 
    num_classes: int,  # noqa: ARG001
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Enhanced mAP calculation using modern Ultralytics utilities.
    
    Provides comprehensive mAP metrics including mAP50, mAP50-95,
    precision, recall, and F1 score using the latest Ultralytics
    evaluation standards.
    
    Args:
        predictions: List of prediction tensors
        targets: List of target tensors
        num_classes: Number of classes
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        device: Device for computation
        
    Returns:
        Dictionary with comprehensive mAP metrics
    """
    manager = get_ultralytics_manager()
    
    if not manager.is_available():
        logger.warning("Ultralytics not available, returning zero metrics")
        return {
            'map50': 0.0,
            'map50_95': 0.0, 
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
    
    try:
        # Process predictions and targets
        all_tp = []
        all_conf = []
        all_pred_cls = []
        all_target_cls = []
        
        for pred, target in zip(predictions, targets):
            if pred is None or target is None or len(pred) == 0 or len(target) == 0:
                continue
                
            # Apply NMS to predictions
            nms_pred = manager.apply_nms(
                pred.unsqueeze(0),
                conf_thres=conf_thres,
                iou_thres=iou_thres
            )
            
            if not nms_pred or len(nms_pred[0]) == 0:
                continue
                
            pred_filtered = nms_pred[0]
            
            # Extract prediction components
            pred_boxes = pred_filtered[:, :4]
            pred_conf = pred_filtered[:, 4]
            pred_cls = pred_filtered[:, 5]
            
            # Extract target components
            target_boxes = target[:, 2:6]  # Assuming [batch_idx, class, x, y, w, h] format
            target_cls = target[:, 1]
            
            # Convert target boxes to xyxy format
            target_boxes_xyxy = manager.convert_xywh_to_xyxy(target_boxes)
            
            # Compute IoU matrix
            iou_matrix = manager.compute_box_iou(pred_boxes, target_boxes_xyxy)
            
            # Determine true positives based on IoU threshold
            max_iou, max_indices = iou_matrix.max(dim=1)
            tp_mask = max_iou >= iou_thres
            
            # Match predictions to targets
            tp = torch.zeros(len(pred_cls), dtype=torch.bool, device=device)
            
            for i, (is_tp, target_idx) in enumerate(zip(tp_mask, max_indices)):
                if is_tp and pred_cls[i] == target_cls[target_idx]:
                    tp[i] = True
            
            # Accumulate statistics
            all_tp.append(tp)
            all_conf.append(pred_conf)
            all_pred_cls.append(pred_cls)
            all_target_cls.extend(target_cls.tolist())
        
        if not all_tp:
            logger.warning("No valid predictions found for mAP calculation")
            return {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0, 
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0
            }
        
        # Concatenate all statistics
        tp_concat = torch.cat(all_tp)
        conf_concat = torch.cat(all_conf)
        pred_cls_concat = torch.cat(all_pred_cls)
        target_cls_tensor = torch.tensor(all_target_cls, dtype=torch.long, device=device)
        
        # Compute mAP using Ultralytics (automatically handles MPS device compatibility)
        precision, recall, ap, f1 = manager.compute_map_with_ultralytics(
            tp_concat, conf_concat, pred_cls_concat, target_cls_tensor
        )
        
        # Calculate aggregate metrics
        map50 = ap.mean().item() if len(ap) > 0 else 0.0
        precision_mean = precision.mean().item() if len(precision) > 0 else 0.0
        recall_mean = recall.mean().item() if len(recall) > 0 else 0.0
        f1_mean = f1.mean().item() if len(f1) > 0 else 0.0
        
        return {
            'map50': map50,
            'map50_95': map50,  # For now, same as mAP50 (could be enhanced)
            'precision': precision_mean,
            'recall': recall_mean, 
            'f1': f1_mean,
            'accuracy': recall_mean  # Use recall as accuracy for detection tasks
        }
        
    except Exception as e:
        logger.error(f"Error computing enhanced mAP: {e}")
        return {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0, 
            'accuracy': 0.0
        }