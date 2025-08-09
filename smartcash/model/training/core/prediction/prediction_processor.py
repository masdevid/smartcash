#!/usr/bin/env python3
"""
Prediction processing for SmartCash YOLO models.

This module handles prediction normalization and format conversion
for SmartCash YOLO models, ensuring consistent input/output formats.
"""

import torch
from typing import Dict, Tuple, Optional

from smartcash.common.logger import get_logger
from .prediction_cache import PredictionCache
from .classification_extractor import ClassificationExtractor
from .target_processor import TargetProcessor

logger = get_logger(__name__)


class PredictionProcessor:
    """Handles prediction format normalization and processing for SmartCash YOLO models.
    
    This processor is optimized for high-performance tensor operations with O(n) complexity
    and implements caching for maximum throughput during training.
    """
    
    def __init__(self, config: Dict, model, model_api=None):
        """
        Initialize prediction processor for SmartCash YOLO models.
        
        Args:
            config: Training configuration dictionary
            model: PyTorch model instance (for backward compatibility)
            model_api: Model API instance that wraps the model (preferred)
        """
        self.config = config
        self.model_api = model_api
        # Store the model for backward compatibility, but prefer model_api
        self.model = model_api.model if model_api is not None else model
        
        # Initialize specialized components with model_api if available, otherwise use model
        self.cache = PredictionCache()
        
        # Get the model to use for processing (prefer model_api)
        processing_model = model_api if model_api is not None else model
        
        # If we have a model_api that has a model attribute, use that
        if hasattr(processing_model, 'model'):
            processing_model = processing_model.model
            
        # Initialize components with the correct model
        self.classification_extractor = ClassificationExtractor(model=processing_model, cache=self.cache)
        self.target_processor = TargetProcessor(cache=self.cache)
        
        # Total classes configuration
        self.total_classes = 17  # Fixed for SmartCashYOLO model
    
    def normalize_training_predictions(self, predictions, phase_num: int, batch_idx: int = 0) -> Dict:
        """
        Normalize training predictions to consistent format.
        
        Args:
            predictions: Raw model predictions
            phase_num: Unused, kept for interface compatibility
            batch_idx: Current batch index (for logging)
            
        Returns:
            Dict with 'predictions' key containing normalized predictions
        """
        return self._normalize_predictions(predictions, batch_idx, "Training")
    
    def normalize_validation_predictions(self, predictions, phase_num: int, batch_idx: int = 0) -> Dict:
        """
        Normalize validation predictions to consistent format.
        
        Args:
            predictions: Raw model predictions
            phase_num: Unused, kept for interface compatibility
            batch_idx: Current batch index (for logging)
            
        Returns:
            Dict with 'predictions' key containing normalized predictions
        """
        return self._normalize_predictions(predictions, batch_idx, "Validation")
    
    def _normalize_predictions(self, predictions, batch_idx: int, context: str) -> Dict:
        """
        Normalize predictions to consistent format for SmartCash YOLO models.
        
        Args:
            predictions: Raw model predictions (tensor, list, or dict)
            batch_idx: Current batch index (for logging)
            context: Context string (Training/Validation)
            
        Returns:
            Dict with 'predictions' key containing normalized predictions
        """
        if batch_idx == 0:
            pred_type = type(predictions).__name__
            if isinstance(predictions, (list, tuple)):
                pred_type += f" (length: {len(predictions)})"
            logger.debug(f"{context} - YOLO model predictions: {pred_type}")
        
        # Handle different prediction formats
        if isinstance(predictions, dict):
            return predictions  # Already in expected format
            
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            # For list/tuple, use first prediction head
            return {'predictions': predictions[0]}
            
        # For tensor or any other format, wrap in dict
        return {'predictions': predictions}
    
    def process_for_metrics(self, predictions: Dict, targets: torch.Tensor, 
                          images: torch.Tensor, device: torch.device) -> Tuple[Dict, Dict]:
        """
        Process predictions and targets for metrics calculation with optimizations.
        
        Args:
            predictions: Normalized predictions dict
            targets: Target tensor in YOLO format
            images: Input images tensor
            device: Device for tensor operations
            
        Returns:
            Tuple of (processed_predictions, processed_targets)
        """
        batch_size = images.shape[0]
        
        # Fast prediction processing with caching
        predictions_tensor = predictions.get('predictions')
        if predictions_tensor is None or predictions_tensor.numel() == 0:
            return self._get_empty_metrics(batch_size, device)
            
        # Use optimized extraction with minimal overhead
        with torch.no_grad():
            layer_output = self.classification_extractor.extract_classification_predictions_fast(
                predictions_tensor, batch_size, device
            )
            
            # Fast target processing
            layer_targets = self.target_processor.extract_target_classes_fast(
                targets, batch_size, device
            )
        
        return {
            'predictions': layer_output
        }, {
            'predictions': layer_targets
        }
    
    def process_for_metrics_lightweight(self, predictions: Dict, targets: torch.Tensor, 
                                      device: torch.device) -> Tuple[Dict, Dict]:
        """
        Lightweight metrics processing for training (optimized version).
        
        Args:
            predictions: Normalized predictions dict (should contain 'predictions' key)
            targets: Target tensor in YOLO format [img_idx, class_id, x1, y1, x2, y2]
            device: Device for tensor operations
            
        Returns:
            Tuple of (processed_predictions, processed_targets) dictionaries
        """
        if 'predictions' not in predictions:
            return self._get_empty_metrics(1, device)
            
        batch_size = targets.shape[0] if targets.numel() > 0 else 1
        
        # Use fast processing methods
        layer_output = self.classification_extractor.extract_classification_predictions_fast(
            predictions['predictions'], batch_size, device
        )
        layer_targets = self.target_processor.extract_target_classes_fast(targets, batch_size, device)
        
        return {
            'predictions': layer_output.detach()
        }, {
            'predictions': layer_targets.detach()
        }
    
    def _get_empty_metrics(self, batch_size: int, device: torch.device) -> Tuple[Dict, Dict]:
        """Return empty metrics tensors for error cases."""
        empty = self.cache.get_cached_zero_tensor(batch_size, device, torch.float)
        return {'predictions': empty}, {'predictions': empty}
    
    def clear_cache(self) -> None:
        """Clear tensor cache to free memory."""
        self.cache.clear_cache()
        # Clear any additional cached attributes
        if hasattr(self, '_last_phase'):
            delattr(self, '_last_phase')
        
    def get_cache_info(self) -> Dict[str, int]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary containing cache statistics
        """
        return self.cache.get_cache_info()
    
    def optimize_for_phase(self, phase_num: int) -> None:
        """
        Optimize processor settings for specific training phase.
        
        Args:
            phase_num: Current training phase number
        """
        # Selective cache clearing (only when necessary)
        if phase_num != getattr(self, '_last_phase', None):
            self.cache.clear_cache()
            self.classification_extractor.optimize_for_phase(phase_num)
            self._last_phase = phase_num
        
        # Update model phase if needed
        if hasattr(self.model, 'current_phase') and self.model.current_phase != phase_num:
            self.model.current_phase = phase_num
            
        logger.debug(f"Optimized prediction processor for phase {phase_num}")
    
    def process_batch_optimized(self, predictions: Dict, targets: torch.Tensor, 
                              batch_idx: int, device: torch.device) -> Tuple[Dict, Dict]:
        """
        Optimized batch processing with reduced overhead.
        
        Args:
            predictions: Model predictions
            targets: Target tensor
            batch_idx: Batch index for caching decisions
            device: Device for operations
            
        Returns:
            Processed predictions and targets
        """
        # Skip expensive processing for some batches during training
        if batch_idx > 0 and batch_idx % 5 != 0:
            # Return cached empty results for non-critical batches
            batch_size = 1
            if 'predictions' in predictions:
                pred_tensor = predictions['predictions']
                if hasattr(pred_tensor, 'shape') and len(pred_tensor.shape) > 0:
                    batch_size = pred_tensor.shape[0]
            
            return self._get_empty_metrics(batch_size, device)
        
        # Full processing for important batches
        return self.process_for_metrics_lightweight(predictions, targets, device)