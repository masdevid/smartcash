#!/usr/bin/env python3
"""
Target processing utilities for prediction processing.

This module handles target tensor processing, filtering, and extraction
for phase-aware training and validation.
"""

import torch
from typing import Optional

from smartcash.common.logger import get_logger
from .prediction_cache import PredictionCache

logger = get_logger(__name__)


class TargetProcessor:
    """Handles processing and extraction of target data."""
    
    def __init__(self, cache: Optional[PredictionCache] = None):
        """Initialize target processor.
        
        Args:
            cache: Prediction cache instance
        """
        self.cache = cache or PredictionCache()
        
        # Phase-aware class filtering (matching loss_manager.py logic)
        self.layer_class_ranges = {
            'layer_1': torch.tensor(range(0, 7), dtype=torch.long),    # Layer 1: Classes 0-6 (denomination detection)
            'layer_2': torch.tensor(range(7, 14), dtype=torch.long),   # Layer 2: Classes 7-13 (l2_* features)
            'layer_3': torch.tensor(range(14, 17), dtype=torch.long),  # Layer 3: Classes 14-16 (l3_* features)
        }
    
    def extract_target_classes(self, targets: torch.Tensor, batch_size: int, device: torch.device, layer_name: str = 'layer_1') -> torch.Tensor:
        """
        Extract target classes for classification metrics with phase-aware filtering.
        
        Time Complexity: O(n) where n is number of targets (vectorized operations)
        Space Complexity: O(batch_size)
        
        Optimized with vectorized operations instead of loops.
        
        Args:
            targets: Target tensor in YOLO format [image_idx, class_id, x, y, w, h]
            batch_size: Batch size
            device: Device for tensor operations
            layer_name: Layer name for phase-aware filtering
            
        Returns:
            Target classes tensor with filtered and remapped class IDs
        """
        return self._extract_target_classes_optimized(targets, batch_size, device, layer_name)
    
    def extract_target_classes_fast(self, targets, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fast target class extraction for training.
        
        Time Complexity: O(n) where n is number of targets
        Space Complexity: O(batch_size)
        """
        try:
            if targets.numel() > 0 and targets.shape[-1] >= 2:
                # Optimized: direct tensor indexing without intermediate variables
                return targets[:, 1].float()
            
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        except Exception:
            logger.debug("Fast target extraction error - using cached fallback")
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
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
        
        Optimized from O(n²) loop to O(n) vectorized operations.
        
        Args:
            targets: Target tensor in YOLO format [image_idx, class_id, x, y, w, h]
            layer_name: Layer name (e.g., 'layer_1')
            
        Returns:
            Filtered targets tensor with remapped class IDs
        """
        if not hasattr(targets, 'shape') or targets.numel() == 0:
            return targets
        
        # Optimized: use cached tensor ranges for better performance
        valid_classes = self.layer_class_ranges.get(layer_name, self.layer_class_ranges['layer_1'])
        
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
