#!/usr/bin/env python3
"""
Classification prediction extraction utilities for SmartCash YOLO.

This module handles extraction and processing of classification predictions
from YOLO model outputs with optimized tensor operations.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from functools import lru_cache

from smartcash.common.logger import get_logger
from .prediction_cache import PredictionCache

logger = get_logger(__name__)


class ClassificationExtractor:
    """Handles extraction of classification predictions from YOLO outputs.
    
    This extractor is optimized for the SmartCashYOLO model architecture
    with a focus on performance and memory efficiency.
    """
    
    def __init__(self, model=None, cache: Optional[PredictionCache] = None):
        """Initialize classification extractor.
        
        Args:
            model: SmartCashYOLO model reference
            cache: Prediction cache instance for reusing tensors
        """
        self.model = model
        self.cache = cache or PredictionCache()
        # SmartCash configuration aligned with custom detection head
        self.num_classes = 17  # Fixed for SmartCash (0-6: main, 7-13: features, 14-16: auth)
        self.output_channels = 66  # 3 anchors * (17 classes + 5 bbox) = 3 * 22 = 66
    
    def extract_classification_predictions(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract classification predictions from YOLO output.
        
        Time Complexity: O(n) where n is batch_size * num_predictions
        Space Complexity: O(batch_size)
        
        Args:
            layer_preds: YOLO model predictions tensor
            batch_size: Number of samples in the batch
            device: Target device for the output tensor
            
        Returns:
            Tensor of shape [batch_size] containing predicted class indices
        """
        if not isinstance(layer_preds, torch.Tensor) or layer_preds.numel() == 0:
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        if layer_preds.dim() == 5:  # YOLOv5 format [batch, anchors, h, w, features]
            return self._extract_from_yolov5(layer_preds, batch_size, device)
            
        return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
    def extract_classification_predictions_fast(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fast classification prediction extraction with parallel processing.
        
        Optimized version using vectorized operations and memory-efficient processing.
        Handles both 4D [batch, 66, h, w] and 5D [batch, anchors, h, w, features] formats.
        
        Args:
            layer_preds: YOLO model predictions tensor 
            batch_size: Number of samples in the batch
            device: Target device for the output tensor
            
        Returns:
            Tensor of shape [batch_size] containing predicted class indices
        """
        try:
            # Handle 4D format from our custom SmartCashDetectionHead
            if layer_preds.dim() == 4 and layer_preds.shape[1] == 66:  # [batch, 66, h, w]
                with torch.no_grad():
                    # Reshape to [batch, 3_anchors, 22_features, h, w]
                    reshaped = layer_preds.view(batch_size, 3, 22, layer_preds.shape[2], layer_preds.shape[3])
                    
                    # Extract class predictions (channels 5:22 = 17 classes) for all anchors
                    class_preds = reshaped[:, :, 5:22, :, :]  # [batch, 3, 17, h, w]
                    
                    # Flatten spatial dimensions: [batch, 3, 17, h*w]
                    class_preds = class_preds.view(batch_size, 3, 17, -1)
                    
                    # Get objectness scores (channel 0) for weighting
                    obj_scores = reshaped[:, :, 0, :, :].view(batch_size, 3, -1)  # [batch, 3, h*w]
                    
                    # Apply sigmoid to objectness and softmax to classes
                    obj_scores = torch.sigmoid(obj_scores)
                    class_probs = F.softmax(class_preds, dim=2)
                    
                    # Weight class probabilities by objectness
                    weighted_probs = class_probs * obj_scores.unsqueeze(2)  # [batch, 3, 17, h*w]
                    
                    # Find best prediction across all anchors and spatial locations
                    weighted_probs = weighted_probs.view(batch_size, -1, 17)  # [batch, 3*h*w, 17]
                    max_probs, max_classes = torch.max(weighted_probs, dim=2)  # [batch, 3*h*w]
                    
                    # Get best prediction per batch item
                    best_pred_idx = torch.argmax(max_probs, dim=1)  # [batch]
                    best_classes = torch.gather(max_classes, 1, best_pred_idx.unsqueeze(1)).squeeze(1)
                    
                    return best_classes.float()
                    
            elif layer_preds.dim() == 5:  # YOLOv5 format [batch, anchors, h, w, features]
                features = layer_preds.shape[-1]
                if features >= 22:  # Has class predictions (5 bbox + 17 classes)
                    # Ultra-fast vectorized operations with memory optimization
                    with torch.no_grad():
                        # Reshape and extract class probabilities in one operation
                        flat_preds = layer_preds.view(batch_size, -1, features)
                        class_probs = flat_preds[..., 5:22]  # Extract 17 class probabilities
                        
                        # Apply softmax for proper probability distribution
                        class_probs = F.softmax(class_probs, dim=-1)
                        
                        # Find max class for each prediction using vectorized operations
                        max_probs, max_classes = torch.max(class_probs, dim=-1)
                        
                        # Get best prediction per batch item (highest confidence)
                        best_pred_idx = torch.argmax(max_probs, dim=1)
                        best_classes = torch.gather(max_classes, 1, best_pred_idx.unsqueeze(1)).squeeze(1)
                        
                        return best_classes.float()
            
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        except Exception as e:
            logger.debug(f"Fast prediction extraction error: {e}")
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
    def _extract_from_yolov5(self, preds: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract class predictions from YOLOv5 output tensor with optimizations.
        
        Args:
            preds: YOLOv5 output tensor [batch, anchors, h, w, features]
            batch_size: Number of samples in the batch
            device: Target device for the output tensor
            
        Returns:
            Tensor of shape [batch_size] containing predicted class indices
        """
        try:
            features = preds.shape[-1]
            if features < 22:  # Need at least 5 bbox + 1 conf + 17 classes
                return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
                
            # Use optimized tensor operations for maximum performance
            with torch.no_grad():
                # Single reshape operation
                flat_preds = preds.view(batch_size, -1, features)
                
                # Extract 17 SmartCash classes (skip bbox + conf)
                class_probs = flat_preds[..., 5:22]  # Fixed 17 classes
                
                # Combined confidence and class probability
                conf_scores = flat_preds[..., 4:5]  # Objectness confidence
                combined_scores = class_probs * conf_scores
                
                # Vectorized max operations
                max_combined, max_indices = torch.max(combined_scores, dim=-1)
                
                # Efficient batch-wise best prediction selection
                best_pred_indices = torch.argmax(max_combined, dim=1)
                best_classes = torch.gather(max_indices, 1, best_pred_indices.unsqueeze(1)).squeeze(1)
                
                return best_classes.float()
            
        except Exception as e:
            logger.warning(f"Error extracting YOLOv5 predictions: {e}")
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
    @lru_cache(maxsize=32)
    def _get_num_classes_for_current_context(self) -> int:
        """Get number of classes for current context (phase and layer).
        
        Cached for performance - avoids repeated model attribute access.
        """
        # SmartCashYOLO uses 17 fine-grained classes during training (aligned with loss.json)
        return 17
    
    def extract_classification_predictions_parallel(self, layer_preds_list: list, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract predictions from multiple layers using parallel processing.
        
        Args:
            layer_preds_list: List of prediction tensors from different layers
            batch_size: Number of samples in the batch
            device: Target device for the output tensor
            
        Returns:
            Tensor of shape [batch_size] containing predicted class indices
        """
        if not layer_preds_list:
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        try:
            # Process multiple layers in parallel using torch.stack
            with torch.no_grad():
                # Stack all layer predictions
                stacked_preds = torch.stack(layer_preds_list, dim=0)  # [num_layers, batch, ...]
                
                # Extract class predictions from all layers simultaneously
                layer_results = []
                for layer_idx in range(len(layer_preds_list)):
                    layer_pred = stacked_preds[layer_idx]
                    layer_result = self.extract_classification_predictions_fast(layer_pred, batch_size, device)
                    layer_results.append(layer_result)
                
                # Combine results from all layers (use mean or voting strategy)
                if layer_results:
                    combined = torch.stack(layer_results, dim=0)
                    # Use most frequent prediction across layers
                    final_preds = torch.mode(combined, dim=0)[0]
                    return final_preds
                    
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        except Exception as e:
            logger.warning(f"Error in parallel prediction extraction: {e}")
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
    def optimize_for_phase(self, phase_num: int) -> None:
        """Optimize extractor for specific training phase.
        
        Args:
            phase_num: Training phase number
        """
        # Clear cache when switching phases to avoid stale data
        self.cache.clear_cache()
        
        # Pre-populate commonly used tensors for the new phase
        if self.model:
            self.model.current_phase = phase_num
            
        logger.debug(f"Optimized classification extractor for phase {phase_num}")
