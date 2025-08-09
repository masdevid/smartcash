#!/usr/bin/env python3
"""
Target processing utilities for prediction processing.

This module now provides a compatibility wrapper around the unified
TargetFormatConverter to maintain backward compatibility while eliminating
code duplication.
"""

import torch
from typing import Optional

from smartcash.common.logger import get_logger
from .prediction_cache import PredictionCache
from .target_format_converter import TargetFormatConverter

logger = get_logger(__name__)


class TargetProcessor:
    """Handles processing and extraction of target data.
    
    This class now serves as a compatibility wrapper around TargetFormatConverter
    to maintain backward compatibility while eliminating code duplication.
    """
    
    def __init__(self, cache: Optional[PredictionCache] = None):
        """Initialize target processor.
        
        Args:
            cache: Prediction cache instance
        """
        self.cache = cache or PredictionCache()
        # Use unified target format converter to eliminate code duplication
        self._target_converter = TargetFormatConverter(cache=self.cache)
        
        # Maintain backward compatibility with layer_class_ranges access
        self.layer_class_ranges = self._target_converter.layer_class_ranges
    
    def extract_target_classes(self, targets: torch.Tensor, batch_size: int, device: torch.device, layer_name: str = 'layer_1') -> torch.Tensor:
        """
        Extract target classes for classification metrics with phase-aware filtering.
        
        Time Complexity: O(n) where n is number of targets (vectorized operations)
        Space Complexity: O(batch_size)
        
        This method now delegates to the unified TargetFormatConverter.
        
        Args:
            targets: Target tensor in YOLO format [image_idx, class_id, x, y, w, h]
            batch_size: Batch size
            device: Device for tensor operations
            layer_name: Layer name for phase-aware filtering
            
        Returns:
            Target classes tensor with filtered and remapped class IDs
        """
        return self._target_converter.extract_target_classes(targets, batch_size, device, layer_name)
    
    def extract_target_classes_fast(self, targets, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fast target class extraction for training.
        
        Time Complexity: O(n) where n is number of targets
        Space Complexity: O(batch_size)
        
        This method now delegates to the unified TargetFormatConverter.
        """
        return self._target_converter.extract_target_classes_fast(targets, batch_size, device)
    
    # Legacy private methods are now handled by TargetFormatConverter
    # These are kept as properties for any external code that may access them
    
    @property
    def _target_converter_instance(self) -> TargetFormatConverter:
        """Access to underlying target converter for advanced use cases."""
        return self._target_converter
