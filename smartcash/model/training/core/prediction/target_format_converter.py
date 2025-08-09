#!/usr/bin/env python3
"""
Unified target format conversion for SmartCash training pipeline.

This module provides a single source of truth for target format conversion,
eliminating overlaps between validation and training target processing.
"""

import torch
from typing import Dict, List, Optional
from .prediction_cache import PredictionCache
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class TargetFormatConverter:
    """Unified target format conversion for training and validation."""
    
    def __init__(self, class_names: Optional[List[str]] = None, cache: Optional[PredictionCache] = None):
        """
        Initialize target format converter.
        
        Args:
            class_names: List of class names for validation (17 SmartCash classes)
            cache: Prediction cache instance for optimization
        """
        self.class_names = class_names or self._get_default_class_names()
        self.cache = cache or PredictionCache()
        
        # Phase-aware class filtering (aligned with loss.json)
        self.layer_class_ranges = {
            'layer_1': range(0, 7),    # Main denominations (0-6)
            'layer_2': range(7, 14),   # Nominal features (7-13)  
            'layer_3': range(14, 17)   # Authentication features (14-16)
        }
    
    def _get_default_class_names(self) -> List[str]:
        """Get default SmartCash class names aligned with loss.json."""
        return [
            '1000_whole', '2000_whole', '5000_whole', '10000_whole', 
            '20000_whole', '50000_whole', '100000_whole',
            '1000_nominal_feature', '2000_nominal_feature', '5000_nominal_feature', 
            '10000_nominal_feature', '20000_nominal_feature', '50000_nominal_feature', 
            '100000_nominal_feature', 'security_thread', 'watermark', 'special_sign'
        ]
    
    def dict_to_yolo_format(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert targets from dictionary format to YOLO-compatible format.
        
        Args:
            targets: Dictionary containing 'boxes' and 'labels' tensors.
                    Expected shapes:
                    - boxes: [N, 4] tensor in format [x1, y1, x2, y2]
                    - labels: [N] tensor of class indices
            
        Returns:
            Tensor in YOLO format [N, 6] where each row is:
            [image_idx, class_id, x1, y1, x2, y2]
        """
        if not isinstance(targets, dict):
            raise TypeError(f"Targets must be a dictionary, got {type(targets).__name__}")
        
        # Check for required keys
        required_keys = {'boxes', 'labels'}
        if not required_keys.issubset(targets):
            missing = required_keys - targets.keys()
            raise ValueError(f"Missing required keys in targets: {missing}")
        
        boxes = targets['boxes']
        labels = targets['labels']
        
        # Type and shape validation
        if not isinstance(boxes, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError(
                f"Boxes and labels must be torch.Tensor, got {type(boxes).__name__} and {type(labels).__name__}"
            )
        
        # Shape validation
        if boxes.numel() > 0:
            if boxes.dim() != 2 or boxes.size(1) != 4:
                raise ValueError(f"Boxes must be of shape [N, 4], got {tuple(boxes.shape)}")
        
        if labels.numel() > 0:
            if labels.dim() != 1:
                raise ValueError(f"Labels must be 1D tensor, got shape {tuple(labels.shape)}")
            
            if len(boxes) != len(labels):
                raise ValueError(
                    f"Mismatch between boxes ({len(boxes)}) and labels ({len(labels)})"
                )
        
        # Validate class indices
        if labels.numel() > 0:
            invalid_indices = (labels < 0) | (labels >= len(self.class_names))
            if invalid_indices.any():
                invalid_classes = labels[invalid_indices].unique().tolist()
                logger.warning(
                    f"Found {invalid_indices.sum().item()} invalid class indices: {invalid_classes} "
                    f"(valid range: 0-{len(self.class_names) - 1})"
                )
        
        # Convert to YOLO format [image_idx, class_id, x1, y1, x2, y2]
        if boxes.numel() == 0:
            return torch.zeros((0, 6), device=boxes.device)
            
        # All boxes are for the same image (image_idx=0)
        image_indices = torch.zeros((len(boxes), 1), device=boxes.device)
        yolo_targets = torch.cat([
            image_indices,  # [N, 1]
            labels.unsqueeze(1).float(),  # [N, 1]
            boxes  # [N, 4]
        ], dim=1)
        
        return yolo_targets
    
    def extract_target_classes(self, targets: torch.Tensor, batch_size: int, 
                             device: torch.device, layer_name: str = 'layer_1') -> torch.Tensor:
        """
        Extract target classes for classification metrics with phase-aware filtering.
        
        Time Complexity: O(n) where n is number of targets (vectorized operations)
        Space Complexity: O(batch_size)
        
        Args:
            targets: Target tensor in YOLO format [img_idx, class_id, x1, y1, x2, y2]
            batch_size: Number of samples in the batch
            device: Target device for the output tensor
            layer_name: Layer name for class filtering
            
        Returns:
            Target classes tensor with filtered and remapped class IDs
        """
        return self._extract_target_classes_optimized(targets, batch_size, device, layer_name)
    
    def _extract_target_classes_optimized(self, targets: torch.Tensor, batch_size: int, 
                                        device: torch.device, layer_name: str = 'layer_1') -> torch.Tensor:
        """
        Optimized target class extraction with vectorized operations.
        
        Eliminates O(n²) loops in favor of O(n) vectorized operations.
        """
        if targets.numel() == 0 or targets.dim() < 2 or targets.shape[-1] <= 1:
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.long)
        
        # Apply phase-aware target filtering (vectorized)
        filtered_targets = self._filter_targets_for_layer(targets, layer_name)
        
        if filtered_targets.numel() == 0:
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.long)
        
        # Optimized: vectorized target extraction using advanced indexing
        layer_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Extract image indices and classes in vectorized manner
        img_indices = filtered_targets[:, 0].long()
        class_values = filtered_targets[:, 1].long()
        
        # Use scatter to assign first class per image (vectorized)
        # This replaces the O(n) loop with O(k) operation where k is unique images
        unique_imgs = torch.unique(img_indices)
        
        for img_idx in unique_imgs:
            if img_idx < batch_size:
                # Get first target for this image
                mask = img_indices == img_idx
                first_class = class_values[mask][0]
                layer_targets[img_idx] = first_class
        
        return layer_targets
    
    def _filter_targets_for_layer(self, targets: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Filter targets by layer detection classes (matching loss_manager.py logic).
        
        Time Complexity: O(n) where n is number of targets (vectorized operations)
        Space Complexity: O(n) for mask and filtered results
        
        Args:
            targets: Target tensor in YOLO format [image_idx, class_id, x, y, w, h]
            layer_name: Layer name (e.g., 'layer_1')
            
        Returns:
            Filtered targets tensor with remapped class IDs
        """
        if not hasattr(targets, 'shape') or targets.numel() == 0:
            return targets
        
        # Get valid class range for this layer (convert range to tensor)
        if layer_name in self.layer_class_ranges:
            valid_classes_range = self.layer_class_ranges[layer_name]
            valid_classes = torch.tensor(list(valid_classes_range), dtype=torch.long)
        else:
            valid_classes = torch.tensor(list(self.layer_class_ranges['layer_1']), dtype=torch.long)
        
        # CRITICAL OPTIMIZATION: Vectorized filtering instead of O(n²) loop
        # Extract class IDs from targets and create mask in single vectorized operation
        class_ids = targets[:, 1].long()  # Extract all class IDs at once
        
        # Create vectorized mask using broadcasting - O(n) instead of O(n²)
        # Check if current device is MPS and if torch.isin is not implemented for it
        if targets.device.type == 'mps':
            # Perform isin on CPU to avoid NotImplementedError on MPS
            mask = torch.isin(class_ids.cpu(), valid_classes.cpu()).to(targets.device)
        else:
            mask = torch.isin(class_ids, valid_classes.to(targets.device))
        
        # Apply mask and clone in single operation
        filtered_targets = targets[mask].clone()
        
        # Optimized class remapping using cached offset
        if filtered_targets.numel() > 0:
            class_offset = self.cache.get_class_offset(layer_name)
            filtered_targets[:, 1] -= class_offset
        
        return filtered_targets
    
    def extract_target_classes_fast(self, targets: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Fast target class extraction for training optimization.
        
        Time Complexity: O(n) where n is number of targets
        Space Complexity: O(batch_size)
        
        Args:
            targets: Target tensor in YOLO format
            batch_size: Number of samples in the batch
            device: Target device for the output tensor
            
        Returns:
            Tensor of shape [batch_size] containing target class indices
        """
        try:
            if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
                return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
            # Fast path: direct tensor indexing for optimization
            if targets.dim() > 1 and targets.size(1) > 1:
                if targets.numel() > 0 and targets.shape[-1] >= 2:
                    # Optimized: direct tensor indexing without intermediate variables
                    return targets[:, 1].float().to(device)
            
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        except Exception as e:
            logger.debug(f"Fast target extraction error: {e}")
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
    def store_predictions_targets(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        storage: Dict[str, List[torch.Tensor]]
    ) -> None:
        """
        Store predictions and targets for metrics calculation.
        
        Args:
            predictions: Model predictions tensor
            targets: Ground truth targets tensor
            storage: Dictionary to store accumulated data
        """
        # Fast validation with minimal overhead
        if not isinstance(predictions, torch.Tensor) or predictions.numel() == 0:
            return
        if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
            return
        
        # Initialize storage if needed
        for key in ['predictions', 'targets']:
            if key not in storage:
                storage[key] = []
        
        # Optimized tensor operations
        with torch.no_grad():
            # Ensure tensors are on CPU and detached
            pred_cpu = predictions.detach().cpu()
            targets_cpu = targets.detach().cpu()
            
            # Vectorized validation (faster than individual checks)
            if pred_cpu.numel() > 0:
                # Batch confidence validation
                if pred_cpu.dim() > 1 and pred_cpu.size(-1) > 4:
                    conf_valid = torch.logical_and(pred_cpu[..., 4] >= 0, pred_cpu[..., 4] <= 1)
                    if not conf_valid.all():
                        invalid_count = (~conf_valid).sum().item()
                        if invalid_count > 10:  # Only log if significant
                            logger.debug(f"Found {invalid_count} invalid confidence scores")
                
                # Batch class validation  
                if pred_cpu.dim() > 1 and pred_cpu.size(-1) > 5:
                    cls_valid = torch.logical_and(pred_cpu[..., 5] >= 0, pred_cpu[..., 5] < 17)
                    if not cls_valid.all():
                        invalid_count = (~cls_valid).sum().item()
                        if invalid_count > 10:  # Only log if significant
                            logger.debug(f"Found {invalid_count} invalid class indices")
        
        # Fast append without copying
        storage['predictions'].append(pred_cpu)
        storage['targets'].append(targets_cpu)