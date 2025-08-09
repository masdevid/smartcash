#!/usr/bin/env python3
"""
Optimized parallel mAP calculator for SmartCash YOLO training.

This module provides high-performance mAP calculation using:
- Vectorized operations
- Parallel class processing
- Memory-efficient tensor operations
- Batch processing optimizations

Aligned with loss.json specifications:
- 17 fine-grained classes during training
- 8 main classes for inference (7 denominations + 1 auth feature)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from multiprocessing import cpu_count
import numpy as np

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ParallelMAPCalculator:
    """High-performance parallel mAP calculator for SmartCash YOLO."""
    
    def __init__(self, num_classes: int = 17, iou_threshold: float = 0.5, 
                 conf_threshold: float = 0.001, use_parallel: bool = True):
        """
        Initialize parallel mAP calculator.
        
        Args:
            num_classes: Number of classes (17 for SmartCash fine-grained)
            iou_threshold: IoU threshold for TP/FP classification
            conf_threshold: Confidence threshold for predictions
            use_parallel: Whether to use parallel processing
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.use_parallel = use_parallel and cpu_count() > 2
        
        # SmartCash class mapping (from loss.json)
        self.fine_to_main = {
            # Main denominations (0-6) -> (0-6)
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
            # Nominal features (7-13) -> corresponding main class (0-6)
            7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6,
            # Authentication features (14-16) -> feature class (7)
            14: 7, 15: 7, 16: 7
        }
        
        # Pre-allocate tensors for performance
        self._initialize_cached_tensors()
        
    def _initialize_cached_tensors(self):
        """Pre-allocate commonly used tensors for performance."""
        self.cached_tensors = {
            'zeros_small': torch.zeros(100, dtype=torch.float32),
            'zeros_medium': torch.zeros(1000, dtype=torch.float32),
            'ones_small': torch.ones(100, dtype=torch.float32),
        }
    
    def compute_batch_iou_vectorized(self, pred_boxes: torch.Tensor, 
                                   target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Vectorized IoU computation for entire batches.
        
        Args:
            pred_boxes: [N, 4] tensor of predicted boxes [x1, y1, x2, y2]
            target_boxes: [M, 4] tensor of target boxes [x1, y1, x2, y2]
            
        Returns:
            [N, M] tensor of IoU values
        """
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return torch.zeros((pred_boxes.size(0), target_boxes.size(0)), dtype=torch.float32)
        
        # Expand dimensions for broadcasting
        pred_boxes = pred_boxes.unsqueeze(1)  # [N, 1, 4]
        target_boxes = target_boxes.unsqueeze(0)  # [1, M, 4]
        
        # Compute intersection coordinates
        x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        
        # Compute intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute areas
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        
        # Compute union and IoU
        union = pred_area + target_area - intersection
        iou = intersection / (union + 1e-6)
        
        return iou.squeeze()
    
    def compute_single_class_ap_fast(self, class_id: int, predictions: List[torch.Tensor], 
                                   targets: List[torch.Tensor]) -> float:
        """
        Fast AP computation for a single class using vectorized operations.
        
        Args:
            class_id: Class ID to compute AP for
            predictions: List of prediction tensors [N, 6] format
            targets: List of target tensors [M, 6] format
            
        Returns:
            Average Precision for the class
        """
        # Collect all predictions and targets for this class
        class_preds = []
        class_targets = []
        
        for pred_tensor, target_tensor in zip(predictions, targets):
            if pred_tensor.numel() == 0 or target_tensor.numel() == 0:
                continue
                
            # Filter predictions by class and confidence
            pred_mask = (pred_tensor[:, 5] == class_id) & (pred_tensor[:, 4] >= self.conf_threshold)
            if pred_mask.any():
                filtered_preds = pred_tensor[pred_mask]
                class_preds.append(filtered_preds)
            
            # Filter targets by class
            target_mask = target_tensor[:, 1] == class_id  # Class is at index 1 for targets
            if target_mask.any():
                filtered_targets = target_tensor[target_mask]
                class_targets.append(filtered_targets)
        
        if not class_preds or not class_targets:
            return 0.0
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(class_preds, dim=0)
        all_targets = torch.cat(class_targets, dim=0)
        
        # Sort predictions by confidence (descending)
        sorted_indices = torch.argsort(all_preds[:, 4], descending=True)
        sorted_preds = all_preds[sorted_indices]
        
        # Vectorized TP/FP computation
        num_targets = len(all_targets)
        tp = torch.zeros(len(sorted_preds), dtype=torch.bool)
        target_matched = torch.zeros(num_targets, dtype=torch.bool)
        
        # Batch IoU computation
        if len(sorted_preds) > 0 and len(all_targets) > 0:
            pred_boxes = sorted_preds[:, :4]  # [x1, y1, x2, y2, conf, cls]
            target_boxes = all_targets[:, 2:6]  # [img_idx, cls, x1, y1, x2, y2]
            
            iou_matrix = self.compute_batch_iou_vectorized(pred_boxes, target_boxes)
            
            # Find best matches for each prediction
            max_ious, max_indices = torch.max(iou_matrix, dim=1)
            
            for i, (max_iou, target_idx) in enumerate(zip(max_ious, max_indices)):
                if max_iou >= self.iou_threshold and not target_matched[target_idx]:
                    tp[i] = True
                    target_matched[target_idx] = True
        
        # Compute precision and recall
        tp_cumsum = torch.cumsum(tp.float(), dim=0)
        fp_cumsum = torch.cumsum((~tp).float(), dim=0)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (num_targets + 1e-6)
        
        # Compute AP using trapezoidal rule
        recall_np = recall.cpu().numpy()
        precision_np = precision.cpu().numpy()
        
        # Add sentinel values
        recall_np = np.concatenate(([0.0], recall_np, [1.0]))
        precision_np = np.concatenate(([1.0], precision_np, [0.0]))
        
        # Compute area under curve
        ap = np.trapz(precision_np, recall_np)
        
        return float(ap)
    
    def compute_map_parallel(self, predictions: List[torch.Tensor], 
                           targets: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute mAP using parallel processing for multiple classes.
        
        Args:
            predictions: List of prediction tensors
            targets: List of target tensors
            
        Returns:
            Dictionary containing mAP metrics
        """
        if not predictions or not targets:
            return {'map': 0.0, 'map50': 0.0}
        
        # Determine if parallel processing is beneficial
        total_predictions = sum(len(p) for p in predictions if p.numel() > 0)
        use_parallel = self.use_parallel and total_predictions > 1000 and self.num_classes > 4
        
        class_aps = {}
        
        if use_parallel:
            # Parallel processing for multiple classes
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, cpu_count())) as executor:
                    # Submit all class computations
                    futures = {
                        executor.submit(self.compute_single_class_ap_fast, class_id, predictions, targets): class_id
                        for class_id in range(self.num_classes)
                    }
                    
                    # Collect results with timeout
                    for future in concurrent.futures.as_completed(futures, timeout=60):
                        class_id = futures[future]
                        try:
                            ap = future.result()
                            class_aps[class_id] = ap
                        except Exception as e:
                            logger.warning(f"AP computation failed for class {class_id}: {e}")
                            class_aps[class_id] = 0.0
                            
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
                use_parallel = False
        
        if not use_parallel:
            # Sequential processing
            for class_id in range(self.num_classes):
                try:
                    ap = self.compute_single_class_ap_fast(class_id, predictions, targets)
                    class_aps[class_id] = ap
                except Exception as e:
                    logger.debug(f"AP computation error for class {class_id}: {e}")
                    class_aps[class_id] = 0.0
        
        # Compute overall metrics
        valid_aps = [ap for ap in class_aps.values() if ap > 0]
        map_score = np.mean(list(class_aps.values())) if class_aps else 0.0
        map50 = map_score  # At IoU=0.5
        
        # Additional metrics
        metrics = {
            'map': float(map_score),
            'map50': float(map50),
            'num_classes': len(class_aps),
            'valid_classes': len(valid_aps)
        }
        
        # Add per-class APs
        for class_id, ap in class_aps.items():
            metrics[f'ap_class_{class_id}'] = float(ap)
        
        logger.debug(f"Computed mAP: {map_score:.4f} over {len(class_aps)} classes")
        
        return metrics
    
    def update_and_compute(self, batch_predictions: torch.Tensor, 
                          batch_targets: torch.Tensor) -> Dict[str, float]:
        """
        Update calculator with a single batch and compute metrics.
        
        Args:
            batch_predictions: Batch predictions tensor [B, N, 6]
            batch_targets: Batch targets tensor [M, 6]
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            # Convert batch to list format
            batch_size = batch_predictions.size(0)
            pred_list = [batch_predictions[i] for i in range(batch_size)]
            target_list = [batch_targets[batch_targets[:, 0] == i] for i in range(batch_size)]
            
            # Compute mAP
            return self.compute_map_parallel(pred_list, target_list)
            
        except Exception as e:
            logger.error(f"Batch mAP computation error: {e}")
            return {'map': 0.0, 'map50': 0.0}
    
    def clear_cache(self):
        """Clear cached tensors to free memory."""
        for tensor in self.cached_tensors.values():
            del tensor
        self._initialize_cached_tensors()


def create_parallel_map_calculator(num_classes: int = 17, 
                                 iou_threshold: float = 0.5,
                                 conf_threshold: float = 0.001) -> ParallelMAPCalculator:
    """Factory function to create optimized mAP calculator."""
    return ParallelMAPCalculator(
        num_classes=num_classes,
        iou_threshold=iou_threshold, 
        conf_threshold=conf_threshold,
        use_parallel=True
    )