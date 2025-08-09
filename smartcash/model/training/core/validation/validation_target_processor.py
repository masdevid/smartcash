#!/usr/bin/env python3
"""
Target processing for validation batch processing.

This module handles conversion and validation of ground truth targets
for validation processing in the SmartCash training pipeline.
"""

import torch
from typing import Dict, List

from smartcash.common.logger import get_logger
from ..prediction.target_format_converter import TargetFormatConverter

logger = get_logger(__name__)


class ValidationTargetProcessor:
    """Handles processing and conversion of validation targets."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize target processor.
        
        Args:
            class_names: List of class names for validation
        """
        self.class_names = class_names
        # Use unified target format converter to eliminate code duplication
        self._target_converter = TargetFormatConverter(class_names=class_names)
    
    def convert_targets_to_yolo_format(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert targets from dictionary format to YOLO-compatible format.
        
        This method now delegates to the unified TargetFormatConverter to eliminate
        code duplication and ensure consistent behavior across the codebase.
        
        Args:
            targets: Dictionary containing 'boxes' and 'labels' tensors.
                    Expected shapes:
                    - boxes: [N, 4] tensor in format [x1, y1, x2, y2]
                    - labels: [N] tensor of class indices
            
        Returns:
            Tensor in YOLO format [N, 6] where each row is:
            [image_idx, class_id, x1, y1, x2, y2]
            
        Raises:
            ValueError: If targets are not in the expected format or contain invalid values
            TypeError: If input types are incorrect
        """
        return self._target_converter.dict_to_yolo_format(targets)
    
    def store_predictions_targets(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        storage: Dict[str, List[torch.Tensor]]
    ) -> None:
        """
        Store predictions and targets for mAP calculation with optimized validation.
        
        This method now delegates to the unified TargetFormatConverter to eliminate
        code duplication and ensure consistent behavior across the codebase.
        
        Args:
            predictions: Model predictions tensor of shape [B, N, 6] where each detection has
                        format [x1, y1, x2, y2, confidence, class_id]
            targets: Ground truth targets tensor of shape [M, 6] where each row has format
                    [image_idx, class_id, x1, y1, x2, y2]
            storage: Dictionary to store accumulated predictions and targets
        """
        return self._target_converter.store_predictions_targets(predictions, targets, storage)