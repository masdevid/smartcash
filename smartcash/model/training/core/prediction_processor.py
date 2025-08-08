#!/usr/bin/env python3
"""
Streamlined prediction processing for the unified training pipeline.

This module handles prediction normalization, format conversion,
and coordinates between specialized processing components.
"""

import torch
from typing import Dict, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.tensor_format_converter import convert_for_yolo_loss
from .prediction_cache import PredictionCache
from .classification_extractor import ClassificationExtractor
from .target_processor import TargetProcessor

logger = get_logger(__name__)


class PredictionProcessor:
    """Handles prediction format normalization and processing.
    
    Optimized for high-performance tensor operations with O(n) complexity.
    Implements caching and vectorization for maximum throughput during training.
    """
    
    def __init__(self, config, model=None):
        """
        Initialize prediction processor.
        
        Args:
            config: Training configuration
            model: Model reference for phase information
        """
        self.config = config
        self.model = model
        
        # Initialize specialized components
        self.cache = PredictionCache()
        self.classification_extractor = ClassificationExtractor(model=model, cache=self.cache)
        self.target_processor = TargetProcessor(cache=self.cache)
        
        # Total classes configuration
        self.total_classes = 17  # Classes 0-16 as per layer_class_ranges
    
    def normalize_training_predictions(self, predictions, phase_num: int, batch_idx: int = 0):
        """
        Normalize training predictions to consistent format.
        
        Time Complexity: O(1) for dict inputs, O(n) for conversions
        Space Complexity: O(n) where n is prediction tensor size
        
        Optimized entry point for training prediction normalization.
        
        Args:
            predictions: Raw model predictions
            phase_num: Current training phase
            batch_idx: Current batch index (for logging)
            
        Returns:
            Normalized predictions in dict format
        """
        return self._normalize_predictions(predictions, phase_num, batch_idx, "Training")
    
    def normalize_validation_predictions(self, predictions, phase_num: int, batch_idx: int = 0):
        """
        Normalize validation predictions to consistent format.
        
        Time Complexity: O(1) for dict inputs, O(n) for conversions
        Space Complexity: O(n) where n is prediction tensor size
        
        Optimized entry point for validation prediction normalization.
        
        Args:
            predictions: Raw model predictions
            phase_num: Current training phase
            batch_idx: Current batch index (for logging)
            
        Returns:
            Normalized predictions in dict format
        """
        return self._normalize_predictions(predictions, phase_num, batch_idx, "Validation")
    
    def _normalize_predictions(self, predictions, phase_num: int, batch_idx: int, context: str):
        """
        Normalize predictions format based on mode and phase.
        
        Time Complexity: O(1) for dict inputs, O(n) for tensor conversion
        Space Complexity: O(k * n) where k is number of layers
        
        Optimized with early returns and cached configuration access.
        
        Args:
            predictions: Raw model predictions
            phase_num: Current training phase
            batch_idx: Current batch index
            context: Context string (Training/Validation)
            
        Returns:
            Normalized predictions in dict format
        """
        # Early return for already normalized predictions
        if isinstance(predictions, dict):
            if batch_idx == 0:
                logger.debug(f"{context} - Model returned dict with layers: {list(predictions.keys())}")
            return predictions
        
        # Cached configuration access for better performance
        current_phase = getattr(self.model if hasattr(self, 'model') else None, 'current_phase', phase_num)
        training_mode = self.config.get('training_mode', 'two_phase')
        layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
        
        # Optimized logging: only on first batch
        if batch_idx == 0:
            logger.debug(f"{context} - Model current_phase: {current_phase}, training_mode: {training_mode}, layer_mode: {layer_mode}")
        
        # Optimized format determination and conversion
        if self._should_use_multi_layer(training_mode, layer_mode, current_phase):
            return self._convert_to_multi_layer_optimized(predictions, batch_idx, context)
        else:
            # Single layer mode: optimized conversion
            if isinstance(predictions, (tuple, list)):
                predictions = convert_for_yolo_loss(predictions)
            return {'layer_1': predictions}
    
    def _should_use_multi_layer(self, training_mode: str, layer_mode: str, current_phase: int) -> bool:
        """Determine if multi-layer format should be used."""
        if training_mode == 'single_phase' and layer_mode == 'multi':
            return True
        elif training_mode == 'two_phase' and current_phase == 2:
            return True
        return False
    
    def _convert_to_multi_layer_optimized(self, predictions, batch_idx: int, context: str):
        """
        Optimized conversion to multi-layer dict format.
        
        Time Complexity: O(1) for tensor sharing, O(n) for conversions
        Space Complexity: O(1) additional (shares tensor references)
        
        Optimized to share tensor references instead of copying data.
        """
        if batch_idx == 0:
            logger.debug(f"Multi-layer {context.lower()}: Converting {type(predictions)} to multi-layer dict")
        
        # Optimized: avoid redundant type checks and use tensor sharing
        if isinstance(predictions, (tuple, list)) and len(predictions) >= 1:
            # Share predictions across layers to save memory
            predictions_dict = {
                'layer_1': predictions,
                'layer_2': predictions,
                'layer_3': predictions
            }
            if batch_idx == 0:
                logger.debug(f"Created multi-layer predictions: {list(predictions_dict.keys())}")
            return predictions_dict
        
        if isinstance(predictions, torch.Tensor):
            # Optimized tensor processing with single conversion
            if predictions.dim() == 3:
                converted_preds = convert_for_yolo_loss(predictions)
                if batch_idx == 0:
                    logger.debug("Converted flattened tensor to multi-layer predictions")
            else:
                converted_preds = predictions
            
            # Share converted predictions across all layers
            return {
                'layer_1': converted_preds,
                'layer_2': converted_preds,
                'layer_3': converted_preds
            }
        
        # Fallback to single layer
        if batch_idx == 0:
            logger.warning(f"Fallback to single layer due to unexpected prediction format: {type(predictions)}")
        return {'layer_1': predictions}
    
    def process_for_metrics(self, predictions: Dict, targets: torch.Tensor, 
                          images: torch.Tensor, device: torch.device) -> Tuple[Dict, Dict]:
        """
        Process predictions and targets for metrics calculation.
        
        Time Complexity: O(k * n) where k is number of layers and n is batch size
        Space Complexity: O(k * n * c) where c is number of classes
        
        Optimized with batch processing and memory-efficient operations.
        
        Args:
            predictions: Normalized predictions dict
            targets: Target tensor
            images: Input images tensor
            device: Device for tensor operations
            
        Returns:
            Tuple of (processed_predictions, processed_targets)
        """
        batch_size = images.shape[0]
        processed_predictions = {}
        processed_targets = {}
        
        # Optimized: process all layers in batch to reduce overhead
        for layer_name, layer_preds in predictions.items():
            # Process predictions with optimized extraction
            layer_output = self.classification_extractor.extract_classification_predictions(
                layer_preds, batch_size, device
            )
            processed_predictions[layer_name] = layer_output.detach()
            
            # Process targets with vectorized filtering
            layer_targets = self.target_processor.extract_target_classes(
                targets, batch_size, device, layer_name
            )
            processed_targets[layer_name] = layer_targets.detach()
        
        return processed_predictions, processed_targets
    
    def process_for_metrics_lightweight(self, predictions: Dict, targets: torch.Tensor, 
                                      device: torch.device) -> Tuple[Dict, Dict]:
        """
        Lightweight metrics processing for training (ultra-fast version).
        
        Time Complexity: O(n) where n is batch size (single layer only)
        Space Complexity: O(n) 
        
        Optimized for maximum training speed - processes only primary layer.
        
        Args:
            predictions: Normalized predictions dict
            targets: Target tensor
            device: Device for tensor operations
            
        Returns:
            Tuple of (processed_predictions, processed_targets)
        """
        # Ultra-optimized: only process layer_1 for training metrics
        if 'layer_1' not in predictions:
            return {}, {}
            
        layer_preds = predictions['layer_1']
        batch_size = targets.shape[0] if targets.numel() > 0 else 1
        
        # Single-pass processing with cached operations
        layer_output = self.classification_extractor.extract_classification_predictions_fast(layer_preds, batch_size, device)
        layer_targets = self.target_processor.extract_target_classes_fast(targets, batch_size, device)
        
        return {
            'layer_1': layer_output.detach()
        }, {
            'layer_1': layer_targets.detach()
        }
    
    # Delegate methods to specialized components
    def extract_classification_predictions(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract classification predictions from YOLO output."""
        return self.classification_extractor.extract_classification_predictions(layer_preds, batch_size, device)
    
    def extract_target_classes(self, targets: torch.Tensor, batch_size: int, device: torch.device, layer_name: str = 'layer_1') -> torch.Tensor:
        """Extract target classes for classification metrics with phase-aware filtering."""
        return self.target_processor.extract_target_classes(targets, batch_size, device, layer_name)
    
    def clear_cache(self) -> None:
        """Clear tensor cache to free memory."""
        self.cache.clear_cache()
        # Clear LRU cache for class count method in classification extractor
        self.classification_extractor._get_num_classes_for_current_context.cache_clear()
        
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        base_info = self.cache.get_cache_info()
        base_info['class_context_cache_info'] = self.classification_extractor._get_num_classes_for_current_context.cache_info()._asdict()
        return base_info
    
    def optimize_for_phase(self, phase_num: int) -> None:
        """Optimize processor settings for specific training phase."""
        # Optimize all components for the new phase
        self.cache.clear_cache()
        self.classification_extractor.optimize_for_phase(phase_num)
        
        # Pre-populate commonly used tensors for the new phase
        if self.model:
            self.model.current_phase = phase_num
            
        logger.debug(f"Optimized prediction processor for phase {phase_num}")