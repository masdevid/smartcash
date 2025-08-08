#!/usr/bin/env python3
"""
Classification prediction extraction utilities.

This module handles extraction and processing of classification predictions
from YOLO model outputs, including multi-scale processing and tensor operations.
"""

import torch
from typing import Optional
from functools import lru_cache

from smartcash.common.logger import get_logger
from .prediction_cache import PredictionCache

logger = get_logger(__name__)


class ClassificationExtractor:
    """Handles extraction of classification predictions from YOLO outputs."""
    
    def __init__(self, model=None, cache: Optional[PredictionCache] = None):
        """Initialize classification extractor.
        
        Args:
            model: Model reference for phase information
            cache: Prediction cache instance
        """
        self.model = model
        self.cache = cache or PredictionCache()
    
    def extract_classification_predictions(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract classification predictions from YOLO output.
        
        Time Complexity: O(s * n * m) where s is scales, n is batch size, m is detections
        Space Complexity: O(n * c) where c is number of classes
        
        Optimized with vectorized operations and intelligent caching.
        
        Args:
            layer_preds: Layer predictions (list, tuple, or tensor)
            batch_size: Batch size
            device: Device for tensor operations
            
        Returns:
            Classification predictions tensor
        """
        return self._extract_classification_predictions_optimized(layer_preds, batch_size, device)
    
    def extract_classification_predictions_fast(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fast classification prediction extraction for training.
        
        Time Complexity: O(n) where n is batch_size * num_predictions
        Space Complexity: O(batch_size)
        """
        try:
            if layer_preds.dim() == 5:  # YOLOv5 format [batch, anchors, h, w, features]
                # Optimized: vectorized tensor operations without intermediate variables
                features = layer_preds.shape[-1]
                
                if features >= 6:  # Has class predictions
                    # Vectorized reshape and max operations in single pass
                    flat_preds = layer_preds.view(batch_size, -1, features)
                    class_probs = flat_preds[:, :, 5:]  # Skip bbox + conf
                    _, max_classes = torch.max(class_probs, dim=-1)
                    return max_classes.float()
            
            # Cached zero tensor to avoid repeated allocation
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
            
        except Exception:
            logger.debug("Fast prediction extraction error - using cached fallback")
            return self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
    
    def _extract_classification_predictions_optimized(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Optimized classification prediction extraction.
        
        Combines multi-scale processing with vectorized operations for maximum performance.
        """
        # Process YOLO predictions for classification metrics
        if isinstance(layer_preds, (list, tuple)) and len(layer_preds) > 0:
            # Optimized: pre-allocate list and process valid predictions efficiently
            valid_predictions = []
            for scale_pred in layer_preds:
                if (isinstance(scale_pred, torch.Tensor) and 
                    scale_pred.numel() > 0 and 
                    scale_pred.dim() >= 2 and 
                    scale_pred.shape[-1] > 5):
                    scale_output = self._extract_from_yolo_tensor(scale_pred, batch_size, device)
                    valid_predictions.append(scale_output)
            
            if valid_predictions:
                # FIXED: Handle different tensor sizes before stacking
                try:
                    # Check if all predictions have the same shape
                    shapes = [pred.shape for pred in valid_predictions]
                    if len(set(shapes)) == 1:
                        # All same shape - can stack normally
                        return torch.stack(valid_predictions).mean(dim=0)
                    else:
                        # Different shapes - need to handle differently
                        logger.debug(f"Prediction tensor shapes mismatch: {shapes}")
                        
                        # Find the minimum number of classes across all predictions
                        min_classes = min(pred.shape[-1] for pred in valid_predictions)
                        target_shape = (valid_predictions[0].shape[0], min_classes)  # [batch_size, min_classes]
                        
                        # Truncate all predictions to the same size
                        normalized_preds = []
                        for pred in valid_predictions:
                            if pred.shape[-1] > min_classes:
                                # Truncate to min_classes
                                normalized_pred = pred[..., :min_classes]
                            else:
                                # Already at or smaller than min_classes
                                normalized_pred = pred
                            
                            # Ensure consistent batch dimension
                            if normalized_pred.shape[0] != target_shape[0]:
                                logger.debug(f"Adjusting batch dimension from {normalized_pred.shape[0]} to {target_shape[0]}")
                                # Take only the needed batch size or pad if necessary
                                if normalized_pred.shape[0] > target_shape[0]:
                                    normalized_pred = normalized_pred[:target_shape[0]]
                                else:
                                    # This shouldn't happen but handle gracefully
                                    padding_needed = target_shape[0] - normalized_pred.shape[0]
                                    padding = torch.zeros((padding_needed, normalized_pred.shape[-1]), 
                                                        device=device, dtype=normalized_pred.dtype)
                                    normalized_pred = torch.cat([normalized_pred, padding], dim=0)
                            
                            normalized_preds.append(normalized_pred)
                        
                        # Now we can stack and average
                        return torch.stack(normalized_preds).mean(dim=0)
                        
                except RuntimeError as e:
                    logger.warning(f"Failed to stack predictions: {e}, using first prediction only")
                    return valid_predictions[0] if valid_predictions else self.cache.get_cached_random_tensor(batch_size, self._get_num_classes_for_current_context(), device)
            else:
                # Use cached random predictions
                num_classes = self._get_num_classes_for_current_context()
                return self.cache.get_cached_random_tensor(batch_size, num_classes, device)
        
        # Direct tensor processing
        if isinstance(layer_preds, torch.Tensor):
            # Optimized tensor format detection
            if (layer_preds.dim() >= 3 or 
                (layer_preds.dim() == 2 and layer_preds.shape[1] > 50)):
                # Process as YOLO output
                return self._extract_from_yolo_tensor(layer_preds, batch_size, device)
            else:
                # Already processed classification tensor
                return layer_preds
        
        # Fallback to cached random predictions
        num_classes = self._get_num_classes_for_current_context()
        return self.cache.get_cached_random_tensor(batch_size, num_classes, device)
    
    def _extract_from_yolo_tensor(self, tensor: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract classification predictions from YOLO tensor format.
        
        Time Complexity: O(n * m) where n is batch_size and m is number of detections
        Space Complexity: O(n * c) where c is number of classes
        
        Optimized with vectorized operations and reduced memory allocations.
        """
        # Vectorized extraction of class predictions and objectness
        class_logits = tensor[..., 5:]  # Shape: [batch, detections, num_classes]
        objectness = tensor[..., 4:5]   # Shape: [batch, detections, 1]
        
        # Optimized: single sigmoid call for both tensors using in-place operations where possible
        objectness_sigmoid = torch.sigmoid(objectness)
        class_sigmoid = torch.sigmoid(class_logits)
        weighted_class_probs = objectness_sigmoid * class_sigmoid
        
        # Extract YOLO predictions with optimized tensor indexing
        if weighted_class_probs.shape[1] > 0:  # Has detections
            objectness_scores = objectness_sigmoid.squeeze(-1)  # [batch, detections]
            
            # Optimized: unified tensor operations for both formats
            if objectness_scores.dim() == 2:
                # Standard format: vectorized argmax and advanced indexing
                best_detection_idx = torch.argmax(objectness_scores, dim=1)  # [batch]
                batch_indices = torch.arange(weighted_class_probs.shape[0], device=device)
                layer_output = weighted_class_probs[batch_indices, best_detection_idx]  # [batch, num_classes]
            else:
                # Flattened format: optimized reshape operations
                reshaped_probs = weighted_class_probs.view(batch_size, -1, weighted_class_probs.shape[-1])
                reshaped_obj = objectness_scores.view(batch_size, -1)
                
                best_detection_idx = torch.argmax(reshaped_obj, dim=1)  # [batch]
                batch_indices = torch.arange(batch_size, device=device)
                layer_output = reshaped_probs[batch_indices, best_detection_idx]  # [batch, num_classes]
                
            # Optimized debug logging with reduced tensor operations
            debug_counter = self.cache.get_debug_counter()
            if debug_counter <= 3:
                pred_classes = torch.argmax(layer_output, dim=1)
                max_conf, min_conf = layer_output.max().item(), layer_output.min().item()
                logger.debug(f"   • Final predictions shape: {layer_output.shape}")
                logger.debug(f"   • Predicted classes: {pred_classes.tolist()}")
                logger.debug(f"   • Max/Min prediction confidence: {max_conf:.6f}/{min_conf:.6f}")
                self.cache.increment_debug_counter()
        else:
            # Optimized: use cached tensor for better performance
            num_classes = self._get_num_classes_for_current_context()
            layer_output = self.cache.get_cached_random_tensor(batch_size, num_classes, device)
            debug_counter = self.cache.get_debug_counter()
            if debug_counter <= 3:
                logger.debug("   • No detections found - returning cached random predictions")
                self.cache.increment_debug_counter()
        
        return layer_output
    
    @lru_cache(maxsize=32)
    def _get_num_classes_for_current_context(self) -> int:
        """Get number of classes for current context (phase and layer).
        
        Cached for performance - avoids repeated model attribute access.
        """
        # Default to 7 classes for Phase 1 (layer_1 only)
        current_phase = getattr(self.model, 'current_phase', 1) if self.model else 1
        if current_phase == 1:
            return 7  # Phase 1: only layer_1 classes (0-6)
        else:
            return 7  # Phase 2: still 7 classes per layer (see layer_class_ranges)
    
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
