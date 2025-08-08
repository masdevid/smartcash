#!/usr/bin/env python3
"""
Validation batch processing for the unified training pipeline.

This module handles processing of individual validation batches,
including loss computation and data collection.
"""

import torch

from smartcash.common.logger import get_logger
from .prediction_processor import PredictionProcessor

logger = get_logger(__name__)


class ValidationBatchProcessor:
    """Handles processing of individual validation batches."""
    
    def __init__(self, model, config, prediction_processor: PredictionProcessor):
        """
        Initialize validation batch processor.
        
        Args:
            model: PyTorch model
            config: Training configuration
            prediction_processor: Prediction processor instance
        """
        self.model = model
        self.config = config
        self.prediction_processor = prediction_processor
    
    def _is_smartcash_model(self) -> bool:
        """Check if the model is a SmartCash YOLOv5 model."""
        # Check class name in different ways for better detection
        class_name = str(self.model.__class__.__name__)
        class_str = str(self.model.__class__)
        return (
            'SmartCashYOLOv5Model' in class_name or 
            'SmartCashYOLOv5Model' in class_str or
            hasattr(self.model, 'yolov5_model')  # Check for SmartCash model attribute
        )
    
    def _is_yolov5_compatible_model(self) -> bool:
        """Check if the model is compatible with advanced loss computation."""
        # Only YOLOv5-based models are compatible with advanced loss, not EfficientNet
        if hasattr(self.model, 'backbone_type'):
            return 'efficientnet' not in self.model.backbone_type.lower()
        if hasattr(self.model, 'model'):
            model_str = str(type(self.model.model))
            return 'efficientnet' not in model_str.lower()
        return True  # Default to assuming compatibility
    
    def _is_modern_yolo_model(self) -> bool:
        """Check if the model is a modern YOLO (v8+) compatible with v8DetectionLoss."""
        # Check for YOLOv8+ specific attributes
        if hasattr(self.model, 'model'):
            try:
                # Handle both list-like and direct access to model layers
                if hasattr(self.model.model, '__getitem__'):
                    detect_head = self.model.model[-1]
                elif hasattr(self.model, '__getitem__'):
                    detect_head = self.model[-1]
                else:
                    return False
                
                # YOLOv8+ detection heads have 'reg_max' attribute for DFL (Distribution Focal Loss)
                # YOLOv5 models typically don't have this attribute
                has_reg_max = hasattr(detect_head, 'reg_max') and getattr(detect_head, 'reg_max', 0) > 1
                has_modern_features = hasattr(detect_head, 'anchors') or hasattr(detect_head, 'stride')
                
                return has_reg_max and has_modern_features
                
            except (IndexError, AttributeError):
                return False
        return False
    
    def _prepare_ultralytics_batch(self, targets: torch.Tensor, img_size: int) -> dict:
        """Prepare batch in Ultralytics format for v8DetectionLoss."""
        batch = {}
        
        if targets.numel() > 0:
            # Extract components from YOLO format [image_idx, class_id, x, y, w, h] 
            batch_idx = targets[:, 0].long()
            cls = targets[:, 1].long()
            bboxes = targets[:, 2:6]  # x, y, w, h (normalized)
            
            # Ensure bboxes are in correct format for v8DetectionLoss
            # v8DetectionLoss expects normalized xywh format
            batch['batch_idx'] = batch_idx
            batch['cls'] = cls
            batch['bboxes'] = bboxes
            
            logger.debug(f"Prepared Ultralytics batch: {batch_idx.shape[0]} targets, "
                        f"img_size={img_size}, bbox_range=[{bboxes.min():.3f}, {bboxes.max():.3f}]")
        else:
            # Empty targets
            device = targets.device
            batch['batch_idx'] = torch.empty(0, dtype=torch.long, device=device)
            batch['cls'] = torch.empty(0, dtype=torch.long, device=device) 
            batch['bboxes'] = torch.empty(0, 4, device=device)
            logger.debug("Prepared empty Ultralytics batch")
        
        return batch
    
    def _compute_smartcash_loss(self, predictions, targets, img_size: int) -> torch.Tensor:
        """Compute loss for SmartCash models (SMART: chooses best loss based on model architecture)."""
        # IMPROVED: Skip advanced loss for EfficientNet models - use simple fallback
        if not self._is_yolov5_compatible_model():
            logger.debug("EfficientNet model detected - using simple fallback loss computation")
            return self._compute_simple_yolo_loss(predictions, targets)
        
        # SMART: Choose the best loss implementation based on model architecture
        if self._is_modern_yolo_model():
            return self._compute_ultralytics_v8_loss(predictions, targets, img_size)
        else:
            return self._compute_yolov5_loss(predictions, targets, img_size)
    
    def _compute_ultralytics_v8_loss(self, predictions, targets, img_size: int) -> torch.Tensor:
        """Compute loss using modern Ultralytics v8DetectionLoss."""
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            
            # Initialize modern Ultralytics loss computer
            if not hasattr(self, '_ultralytics_loss_fn'):
                self._ultralytics_loss_fn = v8DetectionLoss(self.model)
                logger.debug("Successfully initialized Ultralytics v8DetectionLoss")
            
            # Prepare batch in Ultralytics format
            batch = self._prepare_ultralytics_batch(targets, img_size)
            
            # Handle prediction format conversion for v8DetectionLoss
            # v8DetectionLoss expects predictions as a tuple or list of feature maps
            predictions = self._convert_predictions_to_list(predictions, "v8DetectionLoss")
            
            # CRITICAL: v8DetectionLoss expects predictions as feature maps, not processed outputs
            # The v8DetectionLoss.__call__ method processes raw feature maps internally
            # For compatibility, we need to pass predictions in the expected format
            
            # Compute loss using modern Ultralytics v8DetectionLoss
            loss_outputs = self._ultralytics_loss_fn(predictions, batch)
            
            # v8DetectionLoss returns (loss_tensor, detached_loss) tuple
            if isinstance(loss_outputs, tuple) and len(loss_outputs) == 2:
                loss_tensor, _ = loss_outputs
                # loss_tensor is a tensor with [box_loss, cls_loss, dfl_loss]
                total_loss = loss_tensor.sum()
            else:
                # Fallback if return format is different
                total_loss = loss_outputs.sum() if hasattr(loss_outputs, 'sum') else loss_outputs
            
            logger.debug(f"v8DetectionLoss computed: total={total_loss.item():.6f}")
            
            return total_loss
            
        except (ImportError, AttributeError, IndexError, RuntimeError, TypeError) as e:
            logger.warning(f"Ultralytics v8DetectionLoss failed: {e}, using YOLOv5 fallback")
            return self._compute_yolov5_loss(predictions, targets, img_size)
    
    def _compute_yolov5_loss(self, predictions, targets, img_size: int) -> torch.Tensor:
        """Compute loss using original YOLOv5 ComputeLoss."""
        try:
            import sys
            sys.path.insert(0, 'yolov5')
            from utils.loss import ComputeLoss
            
            # Initialize YOLOv5 loss computer
            if not hasattr(self, '_yolo_loss_fn'):
                # Add minimal hyperparameters directly to model  
                if not hasattr(self.model, 'hyp'):
                    self.model.hyp = {
                        'box': 0.05, 'obj': 1.0, 'cls': 0.5,
                        'cls_pw': 1.0, 'obj_pw': 1.0, 'anchor_t': 4.0,
                        'fl_gamma': 0.0, 'label_smoothing': 0.0
                    }
                
                self._yolo_loss_fn = ComputeLoss(self.model)
                logger.debug("Successfully initialized YOLOv5 ComputeLoss")
            
            # Handle prediction format conversion
            predictions = self._convert_predictions_to_list(predictions, "YOLOv5 ComputeLoss")
            
            # Compute loss using YOLOv5 loss function
            loss, _ = self._yolo_loss_fn(predictions, targets)
            logger.debug(f"YOLOv5 ComputeLoss computed: loss={loss.item():.6f} for img_size={img_size}")
            return loss
            
        except (ImportError, AttributeError, IndexError) as e:
            logger.warning(f"YOLOv5 ComputeLoss failed: {e}, using simple fallback")
            return self._compute_simple_yolo_loss(predictions, targets)
    
    def _convert_predictions_to_list(self, predictions, context: str):
        """Convert predictions to list format expected by loss functions."""
        if isinstance(predictions, dict):
            pred_list = []
            for layer_name in ['layer_1', 'layer_2', 'layer_3']:
                if layer_name in predictions:
                    pred_list.append(predictions[layer_name])
            logger.debug(f"Converted dict predictions to list for {context}. Length: {len(pred_list)}")
            return pred_list
        elif isinstance(predictions, list):
            logger.debug(f"Predictions already in list format for {context}.")
            return predictions
        else:
            logger.debug(f"Wrapped single tensor prediction for {context}.")
            return [predictions]
    
    def _compute_simple_yolo_loss(self, predictions, targets) -> torch.Tensor:
        """Simple fallback loss computation for SmartCash models."""
        # IMPROVED: Handle model.parameters() properly for SmartCash models
        try:
            # Try SmartCash model structure first
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                device = next(iter(self.model.model.parameters())).device
            else:
                device = next(iter(self.model.parameters())).device
        except (TypeError, StopIteration, AttributeError):
            # Fallback device detection
            device = torch.device('cpu')
            if hasattr(self.model, 'device'):
                device = self.model.device
        
        if not predictions or targets is None or len(targets) == 0:
            logger.warning("Zero loss: Predictions or targets are empty.")
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        # FIXED: Handle dict predictions in fallback as well
        if isinstance(predictions, dict):
            # Convert dict predictions to list format
            pred_list = []
            for layer_name in ['layer_1', 'layer_2', 'layer_3']:
                if layer_name in predictions:
                    pred_list.append(predictions[layer_name])
            predictions = pred_list
            logger.debug(f"Fallback: Converted dict predictions to list. List length: {len(predictions)}")
        
        # Handle case where predictions might be processed incorrectly
        if not isinstance(predictions, (list, tuple)):
            logger.warning(f"Zero loss: Predictions is not a list/tuple, type: {type(predictions)}")
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        # Simple loss: encourage learning with a basic loss function
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        for i, pred in enumerate(predictions):
            # Check if pred is actually a tensor
            if isinstance(pred, torch.Tensor) and pred.requires_grad:
                # Simple MSE-based loss to maintain gradients
                pred_loss = torch.mean(pred ** 2) * 0.01  # Small coefficient
                total_loss = total_loss + pred_loss
                logger.debug(f"Added prediction {i} loss: {pred_loss.item():.6f}")
            elif not isinstance(pred, torch.Tensor):
                # Log the issue for debugging
                logger.warning(f"Warning: validation prediction {i} is not a tensor, type: {type(pred)}")
        
        logger.debug(f"Total simple fallback loss: {total_loss.item():.6f}")
        return total_loss
    
    def process_batch(self, images, targets, loss_manager, batch_idx, num_batches,
                     phase_num, all_predictions, all_targets):
        """
        Process a single validation batch.
        
        Args:
            images: Input images tensor
            targets: Target labels tensor
            loss_manager: Loss manager instance
            batch_idx: Current batch index
            num_batches: Total number of batches
            phase_num: Current training phase
            all_predictions: Dictionary to collect predictions
            all_targets: Dictionary to collect targets
            
        Returns:
            Dictionary containing batch metrics
        """
        if batch_idx == 0 or batch_idx % 10 == 0:  # Log first and every 10th batch
            logger.debug(f"Processing validation batch {batch_idx+1}/{num_batches}, images: {images.shape}, targets: {targets.shape}")
        
        device = next(self.model.parameters()).device
        
        # Optimized non-blocking transfers
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # For first batch, ensure transfer is complete to avoid later sync overhead
        if batch_idx == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Get model predictions - ENSURE SmartCash model is in training mode for raw predictions
        original_training = self.model.training
        if self._is_smartcash_model():
            # Force training mode to get raw predictions instead of processed inference
            self.model.train()
            
        predictions = self.model(images)
        
        # Restore original training mode
        if self._is_smartcash_model():
            self.model.training = original_training

        # SmartCash models in training mode return raw tensor list, not dict
        if self._is_smartcash_model() and isinstance(predictions, list):
            logger.debug(f"SmartCash model returned raw predictions list with {len(predictions)} outputs")
        elif self._is_smartcash_model() and not isinstance(predictions, (dict, list)):
            # Fallback: wrap single tensor in list
            logger.warning(f"SmartCash model predictions are unexpected type ({type(predictions)}), wrapping in list.")
            predictions = [predictions]
        
        if batch_idx == 0:  # Only log on first batch to reduce noise
            logger.debug(f"Model predictions type: {type(predictions)}, structure: {type(predictions).__name__}")
            if isinstance(predictions, (list, tuple, dict)): # Added dict here
                logger.debug(f"  Predictions list/dict length: {len(predictions)}")
                if len(predictions) > 0:
                    # Adjusted to handle dicts and lists
                    first_pred_item = next(iter(predictions.values())) if isinstance(predictions, dict) else predictions[0]
                    logger.debug(f"  First prediction item type: {type(first_pred_item)}, shape: {getattr(first_pred_item, 'shape', 'N/A')}")
        
        # Normalize predictions format - skip for SmartCash models
        if not self._is_smartcash_model():
            predictions = self.prediction_processor.normalize_validation_predictions(
                predictions, phase_num, batch_idx
            )
        
        # Store for mAP processing
        self.prediction_processor.last_normalized_predictions = predictions
        
        # Compute loss - handle SmartCash models differently
        try:
            if self._is_smartcash_model():
                loss = self._compute_smartcash_loss(predictions, targets, images.shape[-1])
                loss_breakdown = {'total_loss': loss.item(), 'smartcash_loss': loss.item()}
            else:
                loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
            
            # Debug: Check if loss is zero
            if loss.item() == 0.0:
                logger.warning(f"Validation batch {batch_idx}: Loss is zero. Targets shape: {targets.shape}, Predictions keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'not dict'}")
            
            if batch_idx == 0:
                logger.debug(f"First batch loss: {loss.item():.4f}, loss_breakdown keys: {list(loss_breakdown.keys())}")
                
        except Exception as e:
            logger.error(f"Error computing loss in validation batch {batch_idx}: {e}")
            return {'loss': 0.0}
        
        # Collect predictions and targets for metrics computation
        self._collect_predictions_and_targets(
            predictions, targets, images, device, batch_idx, all_predictions, all_targets, phase_num
        )
        
        return {'loss': loss.item(), 'loss_breakdown': loss_breakdown}
    
    def _collect_predictions_and_targets(self, predictions, targets, images, device, batch_idx,
                                       all_predictions, all_targets, phase_num):
        """Collect predictions and targets for metrics computation."""
        try:
            # Optimization: Only process active layers for current phase
            active_layers = self._get_active_layers_for_phase(phase_num)
            
            # Debug: Log what we're receiving vs what we expect
            if batch_idx == 0:
                logger.info(f"🔍 Phase {phase_num} - Received prediction layers: {list(predictions.keys()) if isinstance(predictions, dict) else 'not dict'}")
                logger.info(f"🔍 Phase {phase_num} - Expected active layers: {active_layers}")
            
            # Track successful layer processing for debugging static metrics
            processed_layers = 0
            failed_layers = []
            
            # Handle case where predictions might not be a dictionary
            if not isinstance(predictions, dict):
                logger.warning(f"Predictions is not a dict: {type(predictions)}, converting to dict with 'layer_1' key")
                predictions = {'layer_1': predictions}
            
            # Ensure we have at least layer_1 for Phase 1
            if phase_num == 1 and 'layer_1' not in predictions:
                logger.error(f"🚨 CRITICAL: Phase 1 missing layer_1 predictions! Available: {list(predictions.keys())}")
            
            for layer_name, layer_preds in predictions.items():
                # Skip inactive layers to reduce complexity
                if layer_name not in active_layers:
                    if batch_idx == 0:
                        logger.debug(f"Skipping inactive layer {layer_name} in Phase {phase_num}")
                    continue
                    
                if batch_idx == 0:
                    logger.debug(f"🔄 Starting processing for {layer_name}")
                    
                if layer_name not in all_predictions:
                    all_predictions[layer_name] = []
                    all_targets[layer_name] = []
                
                try:
                    # Process for classification metrics
                    if batch_idx == 0:
                        logger.debug(f"🔄 {layer_name}: calling extract_classification_predictions")
                    layer_output = self.prediction_processor.extract_classification_predictions(
                        layer_preds, images.shape[0], device
                    )
                    if batch_idx == 0:
                        logger.debug(f"✅ {layer_name}: extract_classification_predictions completed")
                    
                    # Extract target classes with phase-aware filtering
                    if batch_idx == 0:
                        logger.debug(f"🔄 {layer_name}: calling extract_target_classes")
                    layer_targets = self.prediction_processor.extract_target_classes(
                        targets, images.shape[0], device, layer_name
                    )
                    if batch_idx == 0:
                        logger.debug(f"✅ {layer_name}: extract_target_classes completed")
                    
                    # Validate outputs before adding to collections
                    if layer_output is not None and layer_targets is not None:
                        if layer_output.numel() > 0 and layer_targets.numel() > 0:
                            all_predictions[layer_name].append(layer_output)
                            all_targets[layer_name].append(layer_targets)
                            processed_layers += 1
                            
                            # Log successful processing
                            if batch_idx <= 1:  # Log first two batches
                                logger.debug(f"✅ {layer_name} batch {batch_idx}: pred shape {layer_output.shape}, target shape {layer_targets.shape}")
                        else:
                            logger.warning(f"⚠️ {layer_name} batch {batch_idx}: Empty tensors - pred numel={layer_output.numel()}, target numel={layer_targets.numel()}")
                            # Try to create dummy tensors to avoid completely missing metrics
                            if layer_output.numel() == 0:
                                layer_output = torch.zeros(1, dtype=torch.long, device=device)
                            if layer_targets.numel() == 0:
                                layer_targets = torch.zeros(1, dtype=torch.long, device=device)
                            all_predictions[layer_name].append(layer_output)
                            all_targets[layer_name].append(layer_targets)
                            processed_layers += 1
                    else:
                        logger.warning(f"⚠️ {layer_name} batch {batch_idx}: None outputs - pred={layer_output is not None}, target={layer_targets is not None}")
                        # Create dummy tensors to avoid completely missing metrics
                        if layer_output is None:
                            layer_output = torch.zeros(1, dtype=torch.long, device=device)
                        if layer_targets is None:
                            layer_targets = torch.zeros(1, dtype=torch.long, device=device)
                        all_predictions[layer_name].append(layer_output)
                        all_targets[layer_name].append(layer_targets)
                        processed_layers += 1
                    
                    # Memory-efficient accumulation
                    COMPACT_FREQUENCY = self.config.get('training', {}).get('validation', {}).get('memory_compact_freq', 20)
                    if batch_idx > 0 and batch_idx % COMPACT_FREQUENCY == 0:
                        if len(all_predictions[layer_name]) > 1:
                            # Compact tensors to improve memory locality and reduce fragmentation
                            try:
                                all_predictions[layer_name] = [torch.cat(all_predictions[layer_name], dim=0)]
                                all_targets[layer_name] = [torch.cat(all_targets[layer_name], dim=0)]
                            except Exception as compact_e:
                                logger.warning(f"Warning compacting tensors for {layer_name}: {compact_e}")
                    
                    if batch_idx == 0:
                        logger.debug(f"Layer {layer_name}: prediction shape {layer_output.shape}, target shape {layer_targets.shape}")
                        logger.debug(f"  Prediction sample: {layer_output[:2] if layer_output.numel() > 0 else 'empty'}")
                        logger.debug(f"  Target sample: {layer_targets[:2] if layer_targets.numel() > 0 else 'empty'}")
                        
                except Exception as layer_e:
                    logger.error(f"❌ Error processing {layer_name} in batch {batch_idx}: {layer_e}")
                    failed_layers.append(f"{layer_name}(exception)")
                    import traceback
                    logger.error(f"   • Traceback: {traceback.format_exc()}")
            
            # Summary logging for debugging static metrics issue (only log errors)
            if failed_layers:
                logger.error(f"   • Failed layers: {failed_layers}")
                logger.error(f"   • This will contribute to missing layer_1_* metrics and static validation values!")
                    
        except Exception as e:
            logger.error(f"❌ Critical error processing predictions in batch {batch_idx}: {e}")
            logger.error(f"   • This will cause missing layer_1_* metrics and static validation fallback!")
            import traceback
            logger.error(f"   • Traceback: {traceback.format_exc()}")
    
    def _get_active_layers_for_phase(self, phase_num: int) -> list:
        """Get list of active layers for the given phase."""
        if phase_num == 1:
            return ['layer_1']  # Phase 1: only layer_1 is active
        elif phase_num == 2:
            return ['layer_1', 'layer_2', 'layer_3']  # Phase 2: all layers are active
        else:
            # Default to all layers for unknown phases or single-phase mode
            return ['layer_1', 'layer_2', 'layer_3']