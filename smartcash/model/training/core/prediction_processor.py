#!/usr/bin/env python3
"""
Prediction processing for the unified training pipeline.

This module handles prediction normalization, format conversion,
and extraction of classification/detection information.
"""

import torch
from typing import Dict, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.tensor_format_converter import convert_for_yolo_loss, ensure_yolo_format

logger = get_logger(__name__)


class PredictionProcessor:
    """Handles prediction format normalization and processing."""
    
    def __init__(self, config, model=None):
        """
        Initialize prediction processor.
        
        Args:
            config: Training configuration
            model: Model reference for phase information
        """
        self.config = config
        self.model = model
    
    def normalize_training_predictions(self, predictions, phase_num: int, batch_idx: int = 0):
        """
        Normalize training predictions to consistent format.
        
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
        
        Args:
            predictions: Raw model predictions
            phase_num: Current training phase
            batch_idx: Current batch index
            context: Context string (Training/Validation)
            
        Returns:
            Normalized predictions in dict format
        """
        if isinstance(predictions, dict):
            return predictions
        
        current_phase = getattr(self.model if hasattr(self, 'model') else None, 'current_phase', phase_num)
        training_mode = self.config.get('training_mode', 'two_phase')
        layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
        
        if batch_idx == 0:  # Only log on first batch
            logger.info(f"{context} - Model current_phase: {current_phase}, training_mode: {training_mode}, layer_mode: {layer_mode}, prediction type: {type(predictions)}")
            logger.info(f"{context} - phase_num parameter: {phase_num}, model.current_phase: {getattr(self.model if hasattr(self, 'model') else None, 'current_phase', 'NOT_SET')}")
        
        # Check if predictions are already in dict format from phase-aware model
        if isinstance(predictions, dict):
            # Model has already returned phase-appropriate predictions
            if batch_idx == 0:
                logger.info(f"{context} - Model returned dict with layers: {list(predictions.keys())}")
            return predictions
        
        # Determine if we should use multi-layer format
        use_multi_layer = self._should_use_multi_layer(training_mode, layer_mode, current_phase)
        
        if use_multi_layer:
            return self._convert_to_multi_layer(predictions, batch_idx, context)
        else:
            # Single layer mode: normalize to layer_1 as expected
            return {'layer_1': predictions}
    
    def _should_use_multi_layer(self, training_mode: str, layer_mode: str, current_phase: int) -> bool:
        """Determine if multi-layer format should be used."""
        if training_mode == 'single_phase' and layer_mode == 'multi':
            return True
        elif training_mode == 'two_phase' and current_phase == 2:
            return True
        return False
    
    def _convert_to_multi_layer(self, predictions, batch_idx: int, context: str):
        """Convert predictions to multi-layer dict format."""
        if batch_idx == 0:
            logger.info(f"Multi-layer {context.lower()}: Converting {type(predictions)} to multi-layer dict")
        
        if isinstance(predictions, (tuple, list)) and len(predictions) >= 1:
            # YOLOv5 returns tuple/list of predictions for different scales
            # For multi-layer training, we use the predictions for all layers
            predictions_dict = {
                'layer_1': predictions,  # Use all scales for layer_1
                'layer_2': predictions,  # Use all scales for layer_2  
                'layer_3': predictions   # Use all scales for layer_3
            }
            if batch_idx == 0:
                logger.info(f"Created multi-layer predictions: {list(predictions_dict.keys())}")
            return predictions_dict
        elif isinstance(predictions, torch.Tensor):
            # Single tensor - might be flattened format
            if predictions.dim() == 3:
                # Likely flattened format [batch, total_predictions, features]
                # Convert to proper YOLO format for each layer
                converted_preds = convert_for_yolo_loss(predictions)
                predictions_dict = {
                    'layer_1': converted_preds,
                    'layer_2': converted_preds,
                    'layer_3': converted_preds
                }
                if batch_idx == 0:
                    logger.info(f"Converted flattened tensor to multi-layer predictions: {list(predictions_dict.keys())}")
                return predictions_dict
            else:
                # Other tensor format - use as is
                predictions_dict = {
                    'layer_1': predictions,
                    'layer_2': predictions,
                    'layer_3': predictions
                }
                return predictions_dict
        else:
            # Fallback to single layer
            if batch_idx == 0:
                logger.warning(f"Fallback to single layer due to unexpected prediction format: {type(predictions)}")
            return {'layer_1': predictions}
    
    def process_for_metrics(self, predictions: Dict, targets: torch.Tensor, 
                          images: torch.Tensor, device: torch.device) -> Tuple[Dict, Dict]:
        """
        Process predictions and targets for metrics calculation.
        
        Args:
            predictions: Normalized predictions dict
            targets: Target tensor
            images: Input images tensor
            device: Device for tensor operations
            
        Returns:
            Tuple of (processed_predictions, processed_targets)
        """
        processed_predictions = {}
        processed_targets = {}
        
        for layer_name, layer_preds in predictions.items():
            # Process predictions
            layer_output = self.extract_classification_predictions(layer_preds, images.shape[0], device)
            processed_predictions[layer_name] = layer_output.detach()
            
            # Process targets
            layer_targets = self.extract_target_classes(targets, images.shape[0], device)
            processed_targets[layer_name] = layer_targets.detach()
        
        return processed_predictions, processed_targets
    
    def extract_classification_predictions(self, layer_preds, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract classification predictions from YOLO output.
        
        Args:
            layer_preds: Layer predictions (list, tuple, or tensor)
            batch_size: Batch size
            device: Device for tensor operations
            
        Returns:
            Classification predictions tensor
        """
        # Process YOLO predictions for classification metrics
        if isinstance(layer_preds, list) and len(layer_preds) > 0:
            # Extract classification predictions from YOLO output
            first_scale = layer_preds[0]
            if isinstance(first_scale, torch.Tensor) and first_scale.numel() > 0:
                if first_scale.dim() >= 2 and first_scale.shape[-1] > 5:
                    return self._extract_from_yolo_tensor(first_scale, batch_size, device)
                else:
                    return torch.zeros((batch_size, 7), device=device)
            else:
                return torch.zeros((batch_size, 7), device=device)
        else:
            # Direct tensor - assume it's already in the right format
            if isinstance(layer_preds, torch.Tensor):
                return layer_preds
            else:
                return torch.zeros((batch_size, 7), device=device)
    
    def _extract_from_yolo_tensor(self, tensor: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract classification predictions from YOLO tensor format."""
        # Extract class predictions (after bbox and objectness)
        class_logits = tensor[..., 5:]  # Shape: [batch, detections, num_classes]
        objectness = tensor[..., 4:5]   # Shape: [batch, detections, 1]
        
        # Apply objectness weighting to class probabilities
        weighted_class_probs = torch.sigmoid(objectness) * torch.sigmoid(class_logits)
        
        # Debug: Check for valid predictions
        max_obj = torch.sigmoid(objectness).max().item() if objectness.numel() > 0 else 0.0
        max_class = torch.sigmoid(class_logits).max().item() if class_logits.numel() > 0 else 0.0
        
        # Take the detection with highest objectness score per image
        if weighted_class_probs.shape[1] > 0:  # Has detections
            objectness_scores = torch.sigmoid(objectness.squeeze(-1))  # [batch, detections]
            
            # Handle different YOLO output formats
            if objectness_scores.dim() == 2:
                # Standard format: [batch, detections]
                best_detection_idx = torch.argmax(objectness_scores, dim=1)  # [batch]
                batch_indices = torch.arange(weighted_class_probs.shape[0], device=device)
                layer_output = weighted_class_probs[batch_indices, best_detection_idx]  # [batch, num_classes]
            else:
                # Flattened format: reshape and take max over spatial dimensions
                reshaped_probs = weighted_class_probs.view(batch_size, -1, weighted_class_probs.shape[-1])
                reshaped_obj = objectness_scores.view(batch_size, -1)
                
                best_detection_idx = torch.argmax(reshaped_obj, dim=1)  # [batch]
                batch_indices = torch.arange(batch_size, device=device)
                layer_output = reshaped_probs[batch_indices, best_detection_idx]  # [batch, num_classes]
        else:
            layer_output = torch.zeros((batch_size, 7), device=device)
        
        return layer_output
    
    def extract_target_classes(self, targets: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Extract target classes for classification metrics.
        
        Args:
            targets: Target tensor in YOLO format [image_idx, class_id, x, y, w, h]
            batch_size: Batch size
            device: Device for tensor operations
            
        Returns:
            Target classes tensor
        """
        if targets.numel() > 0 and targets.dim() >= 2 and targets.shape[-1] > 1:
            layer_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for img_idx in range(batch_size):
                img_targets = targets[targets[:, 0] == img_idx]  # Filter by image index
                if len(img_targets) > 0:
                    classes = img_targets[:, 1].long()  # Extract class column
                    if len(classes) > 0:
                        layer_targets[img_idx] = classes[0]  # Use first class
        else:
            layer_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        return layer_targets