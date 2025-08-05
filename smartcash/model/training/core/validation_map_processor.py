#!/usr/bin/env python3
"""
Validation mAP processing for the unified training pipeline.

This module handles processing of predictions for mAP calculation,
including format conversion and target preparation.
"""

import torch

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ValidationMapProcessor:
    """Handles mAP-related processing for validation."""
    
    def __init__(self, map_calculator):
        """
        Initialize validation mAP processor.
        
        Args:
            map_calculator: YOLOv5 mAP calculator instance
        """
        self.map_calculator = map_calculator
    
    def update_map_calculator(self, predictions, targets, images, batch_idx, epoch=0):
        """
        Update mAP calculator with batch data for object detection metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            images: Input images
            batch_idx: Current batch index
            epoch: Current epoch number for debug logging
        """
        try:
            # Debug: Log raw prediction data (only when debug logging is enabled)
            if batch_idx == 0:
                logger.debug(f"🔍 RAW PREDICTIONS DEBUG - Batch {batch_idx}:")
                if isinstance(predictions, dict):
                    for key, value in predictions.items():
                        if hasattr(value, 'shape'):
                            logger.debug(f"  • {key}: {value.shape}")
                else:
                    logger.debug(f"  • predictions type: {type(predictions)}")
                    if hasattr(predictions, 'shape'):
                        logger.debug(f"  • predictions shape: {predictions.shape}")
                
                logger.debug(f"  • targets shape: {targets.shape}")
                logger.debug(f"  • targets sample: {targets[:3] if targets.shape[0] > 0 else 'empty'}")
            
            map_predictions = self._prepare_map_predictions(predictions, images.shape[0])
            map_targets = self._prepare_map_targets(targets)
            
            # Debug: Log conversion results (only when debug logging is enabled)
            if batch_idx == 0:
                logger.debug(f"🔄 CONVERSION RESULTS:")
                logger.debug(f"  • map_predictions: {map_predictions.shape if map_predictions is not None else 'None'}")
                logger.debug(f"  • map_targets: {map_targets.shape if map_targets is not None else 'None'}")
                if map_predictions is not None and map_predictions.numel() > 0:
                    logger.debug(f"  • prediction sample: {map_predictions[0, 0] if map_predictions.shape[1] > 0 else 'no detections'}")
                if map_targets is not None and map_targets.numel() > 0:
                    logger.debug(f"  • target sample: {map_targets[0] if map_targets.shape[0] > 0 else 'no targets'}")
            
            if map_predictions is not None and map_targets is not None:
                # Debug: Log detailed mAP data for first batch (only when debug logging is enabled)
                if batch_idx == 0:
                    logger.debug(f"🔍 mAP DEBUG - Batch {batch_idx}:")
                    logger.debug(f"  • Predictions shape: {map_predictions.shape}")
                    logger.debug(f"  • Targets shape: {map_targets.shape}")
                    if map_predictions.numel() > 0:
                        conf_scores = map_predictions[:, :, 4] if map_predictions.dim() == 3 else map_predictions[:, 4]
                        logger.debug(f"  • Confidence range: {conf_scores.min():.6f} - {conf_scores.max():.6f}")
                        logger.debug(f"  • Non-zero confidences: {(conf_scores > 0.01).sum().item()}/{conf_scores.numel()}")
                        logger.debug(f"  • Classes present: {map_predictions[:, :, 5].unique() if map_predictions.dim() == 3 else map_predictions[:, 5].unique()}")
                
                self.map_calculator.update(map_predictions, map_targets, epoch)
                
                if batch_idx == 0:
                    logger.debug(f"mAP data - Predictions: {map_predictions.shape}, Targets: {map_targets.shape}")
            
        except Exception as e:
            logger.warning(f"Error updating mAP calculator for batch {batch_idx}: {e}")
            if batch_idx == 0:  # Only log details for first batch to avoid spam
                logger.debug(f"mAP update error details: predictions type={type(predictions)}, targets shape={targets.shape}")
    
    def _prepare_map_predictions(self, predictions, batch_size=None):
        """
        Convert SmartCash model predictions to YOLOv5 mAP format.
        
        Args:
            predictions: Model predictions (dict or tensor)
            batch_size: Batch size
            
        Returns:
            Tensor in YOLOv5 format [batch, detections, 6] where each detection is [x, y, w, h, conf, class]
        """
        try:
            # For object detection, we primarily use layer_1 predictions (main banknote detection)
            if isinstance(predictions, dict):
                if 'layer_1' in predictions:
                    layer_1_preds = predictions['layer_1']
                else:
                    # Fallback to first available layer
                    first_key = next(iter(predictions.keys()))
                    layer_1_preds = predictions[first_key]
                    logger.debug(f"Using {first_key} predictions for mAP (layer_1 not found)")
            else:
                layer_1_preds = predictions
            
            # Handle different prediction formats
            if isinstance(layer_1_preds, (list, tuple)):
                layer_1_preds = layer_1_preds[0] if len(layer_1_preds) > 0 else None
            
            if layer_1_preds is None:
                return None
            
            # Convert to tensor if not already
            if not isinstance(layer_1_preds, torch.Tensor):
                return None
            
            # Handle YOLOv5 feature map format: [batch, anchors, height, width, features]
            if layer_1_preds.dim() == 5:
                batch_size, num_anchors, height, width, num_features = layer_1_preds.shape
                logger.debug(f"Converting YOLOv5 feature map: {layer_1_preds.shape} -> [batch, detections, features]")
                
                # Reshape to [batch, detections, features] where detections = anchors * height * width
                layer_1_preds = layer_1_preds.permute(0, 1, 2, 3, 4).contiguous()
                layer_1_preds = layer_1_preds.view(batch_size, num_anchors * height * width, num_features)
                
                # Filter out predictions with low confidence
                if num_features >= 5:  # Has confidence score
                    conf_scores = layer_1_preds[:, :, 4]  # Objectness/confidence at index 4
                    conf_threshold = 0.01  # Extra low threshold for early training with new anchors
                    
                    # Keep detections above threshold per batch
                    valid_detections = []
                    for batch_idx in range(batch_size):
                        batch_preds = layer_1_preds[batch_idx]
                        valid_mask = conf_scores[batch_idx] > conf_threshold
                        valid_preds = batch_preds[valid_mask]
                        
                        # Ensure we have some detections even if all are low confidence
                        if valid_preds.shape[0] == 0:
                            # Take top 100 predictions by confidence
                            top_k = min(100, batch_preds.shape[0])
                            _, top_indices = conf_scores[batch_idx].topk(top_k)
                            valid_preds = batch_preds[top_indices]
                        
                        valid_detections.append(valid_preds)
                    
                    # Pad to same length for batch processing
                    max_detections = max(pred.shape[0] for pred in valid_detections)
                    max_detections = min(max_detections, 300)  # Cap at 300 detections per image
                    
                    padded_preds = []
                    for batch_preds in valid_detections:
                        if batch_preds.shape[0] > max_detections:
                            # Take top predictions by confidence
                            conf = batch_preds[:, 4]
                            _, top_indices = conf.topk(max_detections)
                            batch_preds = batch_preds[top_indices]
                        elif batch_preds.shape[0] < max_detections:
                            # Pad with zeros
                            padding = torch.zeros(max_detections - batch_preds.shape[0], batch_preds.shape[1], 
                                                device=batch_preds.device, dtype=batch_preds.dtype)
                            batch_preds = torch.cat([batch_preds, padding], dim=0)
                        padded_preds.append(batch_preds)
                    
                    layer_1_preds = torch.stack(padded_preds, dim=0)
            
            # Ensure we have the right format [batch, detections, features] for other cases
            elif layer_1_preds.dim() == 2:
                # Add batch dimension if missing
                layer_1_preds = layer_1_preds.unsqueeze(0)
            
            elif layer_1_preds.dim() != 3:
                logger.warning(f"Cannot handle prediction tensor dimensions: {layer_1_preds.shape}")
                return None
            
            # Expected format: [batch, detections, features]
            # For YOLO: features should include [x, y, w, h, objectness, class_probs...]
            num_features = layer_1_preds.shape[-1]
            if num_features < 6:  # Need at least x,y,w,h,conf,class
                logger.debug(f"Prediction features too few for mAP: {num_features} < 6")
                return None
            
            # Check if we have class probabilities (features > 6 means we have class probs after objectness)
            if num_features > 6:
                # Format: [x, y, w, h, objectness, class0, class1, ..., classN]
                # Convert to: [x, y, w, h, conf, class_index]
                logger.debug(f"Converting {num_features} features to YOLO format with class probabilities")
                
                coords = layer_1_preds[:, :, :4]  # x, y, w, h
                objectness = layer_1_preds[:, :, 4]  # objectness score
                class_probs = layer_1_preds[:, :, 5:]  # class probabilities
                
                # Get maximum class and its probability
                max_class_prob, max_class = torch.max(class_probs, dim=-1)
                
                # Combine objectness with class probability for final confidence  
                # Clamp to valid range [0, 1] to prevent issues with mAP calculation
                confidence = torch.clamp(objectness * max_class_prob, 0.0, 1.0)
                
                # Create final prediction tensor [x, y, w, h, conf, class]
                map_preds = torch.cat([
                    coords,                      # x, y, w, h
                    confidence.unsqueeze(-1),    # confidence
                    max_class.float().unsqueeze(-1)  # class index
                ], dim=-1)
                
            else:
                # Already in the right format [x, y, w, h, conf, class] 
                map_preds = layer_1_preds.clone()
            
            # Legacy handling (should not be needed with above logic)
            if map_preds.shape[-1] > 6:
                # Take objectness as confidence and get max class
                objectness = map_preds[:, :, 4]  # Objectness score
                class_probs = map_preds[:, :, 5:]  # Class probabilities
                max_class = torch.argmax(class_probs, dim=-1).float()
                
                # Combine objectness with max class probability for confidence
                max_class_prob = torch.max(class_probs, dim=-1)[0]
                confidence = objectness * max_class_prob
                
                # Reconstruct as [x, y, w, h, conf, class]
                map_preds = torch.cat([
                    map_preds[:, :, :4],  # x, y, w, h
                    confidence.unsqueeze(-1),  # conf
                    max_class.unsqueeze(-1)   # class
                ], dim=-1)
            
            # CRITICAL FIX: Normalize coordinates to match target format
            # Predictions are in pixel coordinates, targets are normalized (0-1)
            # We need to normalize predictions to match
            
            # Get input image dimensions from batch_size parameter context
            # Assume standard input size for now - this should be passed as parameter
            input_height, input_width = 640, 640  # Standard YOLO input size
            
            # Normalize coordinates from pixel to 0-1 range
            map_preds[:, :, 0] = map_preds[:, :, 0] / input_width   # x center
            map_preds[:, :, 1] = map_preds[:, :, 1] / input_height  # y center  
            map_preds[:, :, 2] = map_preds[:, :, 2] / input_width   # width
            map_preds[:, :, 3] = map_preds[:, :, 3] / input_height  # height
            
            # Clamp to valid range [0, 1] to prevent out-of-bounds coordinates
            map_preds[:, :, :4] = torch.clamp(map_preds[:, :, :4], 0.0, 1.0)
            
            logger.debug(f"✅ Normalized predictions: coord range {map_preds[:, :, :4].min():.3f}-{map_preds[:, :, :4].max():.3f}")
            
            return map_preds.detach()
            
        except Exception as e:
            logger.warning(f"Error preparing mAP predictions: {e}")
            return None
    
    def _prepare_map_targets(self, targets):
        """
        Convert SmartCash targets to YOLOv5 mAP format.
        
        Args:
            targets: Ground truth targets [N, 6] format: [batch_idx, class, x, y, w, h]
            
        Returns:
            Tensor in YOLOv5 format [num_targets, 6]: [batch_idx, class, x, y, w, h]
        """
        try:
            if targets is None:
                return None
            
            # Convert to tensor if not already
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            
            # Ensure targets are on CPU for mAP calculation
            targets = targets.detach().cpu()
            
            if targets.numel() == 0:
                return torch.empty((0, 6))
            
            # Expected format: [batch_idx, class, x, y, w, h]
            if targets.dim() == 2 and targets.shape[-1] >= 6:
                map_targets = targets[:, :6].clone()  # Take first 6 columns
                
                # CRITICAL FIX: Filter out invalid class indices
                # SmartCash model has 7 classes (0-6), but targets may contain invalid classes (7-13)
                num_classes = 7  # SmartCash banknote classes: 0-6
                valid_mask = map_targets[:, 1] < num_classes  # Class index is at position 1
                
                if valid_mask.sum() == 0:
                    logger.debug("⚠️ No valid target classes found after filtering")
                    return torch.empty((0, 6))
                
                # Filter to only valid classes
                map_targets = map_targets[valid_mask]
                
                logger.debug(f"✅ Filtered targets: {targets.shape[0]} -> {map_targets.shape[0]} (kept classes: {map_targets[:, 1].unique()})")
                
                return map_targets
            
            logger.debug(f"Unexpected target format for mAP: {targets.shape}")
            return None
            
        except Exception as e:
            logger.warning(f"Error preparing mAP targets: {e}")
            return None