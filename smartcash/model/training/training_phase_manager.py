#!/usr/bin/env python3
"""
Training phase management for the unified training pipeline.

This module handles the actual training execution phases including
epoch training, validation, metrics tracking, and phase orchestration.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from torch.amp import autocast

from smartcash.common.logger import get_logger
from smartcash.model.training.data_loader_factory import DataLoaderFactory
from smartcash.model.training.metrics_tracker import MetricsTracker
from smartcash.model.training.optimizer_factory import OptimizerFactory
from smartcash.model.training.loss_manager import LossManager
from smartcash.model.training.utils.early_stopping import create_early_stopping
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics
from smartcash.model.training.utils.checkpoint_utils import generate_checkpoint_name, save_checkpoint_to_disk

logger = get_logger(__name__)


class TrainingPhaseManager:
    """Manages the execution of training phases."""
    
    def __init__(self, model, model_api, config, progress_tracker, 
                 emit_metrics_callback=None, emit_live_chart_callback=None, 
                 visualization_manager=None):
        """
        Initialize training phase manager.
        
        Args:
            model: PyTorch model
            model_api: Model API instance
            config: Training configuration
            progress_tracker: Progress tracking instance
            emit_metrics_callback: Callback for metrics emission
            emit_live_chart_callback: Callback for live chart updates
            visualization_manager: Visualization manager instance
        """
        self.model = model
        self.model_api = model_api
        self.config = config
        self.progress_tracker = progress_tracker
        self.emit_metrics_callback = emit_metrics_callback
        self.emit_live_chart_callback = emit_live_chart_callback
        self.visualization_manager = visualization_manager
        
        # Training state
        self._last_predictions = None
        self._last_targets = None
        self._is_single_phase = False
    
    def run_training_phase(self, phase_num: int, epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Run training phase with specified number of epochs.
        
        Args:
            phase_num: Training phase number (1 or 2)
            epochs: Total number of epochs to train for
            start_epoch: Epoch to start training from (0-based). Defaults to 0.
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            phase_config = self.config['training_phases'][f'phase_{phase_num}']
            data_factory = DataLoaderFactory(self.config)
            train_loader = data_factory.create_train_loader()
            val_loader = data_factory.create_val_loader()
            loss_manager = LossManager(self.config)
            
            # Set current phase on model for layer mode control
            self._set_model_phase(phase_num)
            
            # Configure layer mode based on phase and configuration
            training_mode = self.config.get('training_mode', 'two_phase')
            
            if training_mode == 'single_phase':
                # Single-phase mode: respect the configured layer mode
                single_layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
                if single_layer_mode == 'single':
                    self.model.force_single_layer = True
                    logger.info(f"🎯 Single-phase mode: forcing single layer output")
                else:
                    self.model.force_single_layer = False
                    logger.info(f"🎯 Single-phase mode: using multi-layer output")
            elif training_mode == 'two_phase':
                # Two-phase mode: Phase 1 = single layer, Phase 2 = multi layer
                if phase_num == 1:
                    logger.info(f"🎯 Phase {phase_num}: single layer mode (layer_1 only)")
                else:
                    logger.info(f"🎯 Phase {phase_num}: multi-layer mode (all layers)")
                self.model.force_single_layer = False  # Use phase-based logic
            
            # Create optimizer using OptimizerFactory
            optimizer_factory = OptimizerFactory(self.config)
            base_lr = phase_config.get('learning_rate', 0.001)
            optimizer = optimizer_factory.create_optimizer(self.model, base_lr)
            scheduler = optimizer_factory.create_scheduler(optimizer, total_epochs=epochs)
            scaler = torch.amp.GradScaler('cuda') if self.config['training']['mixed_precision'] else None
            
            # Initialize MetricsTracker
            metrics_tracker = MetricsTracker(config=self.config)
            
            # Create phase-specific early stopping configuration
            es_config = self.config['training']['early_stopping'].copy()
            
            # Check if phase-specific early stopping is enabled
            # For two-phase mode, Phase 1 should have early stopping disabled by default
            training_mode = self.config.get('training_mode', 'single_phase')
            if training_mode == 'two_phase':
                phase_1_enabled = es_config.get('phase_1_enabled', False)  # Disabled by default for Phase 1
                phase_2_enabled = es_config.get('phase_2_enabled', True)   # Enabled by default for Phase 2
            else:
                phase_1_enabled = es_config.get('phase_1_enabled', True)   # Single-phase respects config
                phase_2_enabled = es_config.get('phase_2_enabled', True)
            
            # For two-phase mode, apply phase-specific early stopping logic
            if self.config.get('training_mode') == 'two_phase':
                if phase_num == 1:
                    es_config['enabled'] = phase_1_enabled
                    if not phase_1_enabled:
                        logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
                elif phase_num == 2:
                    es_config['enabled'] = phase_2_enabled
                    if phase_2_enabled:
                        logger.info(f"✅ Early stopping enabled for Phase {phase_num}")
                    else:
                        logger.info(f"🚫 Early stopping disabled for Phase {phase_num}")
            
            early_stopping = create_early_stopping({'training': {'early_stopping': es_config}})
            logger.info(f"🔍 Early stopping object type: {type(early_stopping).__name__} | Config enabled: {es_config.get('enabled', True)}")
            
            # Initialize variables for tracking best model
            best_metrics = {}
            best_checkpoint_path = None
            
            # Start epoch tracking
            self.progress_tracker.start_epoch_tracking(epochs)
            
            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                
                # Update epoch progress
                self.progress_tracker.update_epoch_progress(epoch, epochs, f"Training epoch {epoch + 1}/{epochs}")
                
                # Training
                train_metrics = self._train_epoch(train_loader, optimizer, loss_manager, scaler, epoch, epochs, phase_num)
                
                # Validation
                val_metrics = self._validate_epoch(val_loader, loss_manager, epoch, epochs, phase_num)
                
                # Combine metrics efficiently
                final_metrics = {**train_metrics, **val_metrics}
                
                # Add metrics tracker computed metrics
                tracker_metrics = metrics_tracker.compute_epoch_metrics(epoch)
                final_metrics.update(tracker_metrics)
                
                # Calculate layer metrics efficiently
                layer_metrics = {}
                if hasattr(self, '_last_predictions') and hasattr(self, '_last_targets'):
                    layer_metrics = calculate_multilayer_metrics(self._last_predictions, self._last_targets)
                    final_metrics.update(layer_metrics)
                
                # Ensure accuracy, precision, recall, f1 are always included (even if zero)
                for layer in ['layer_1', 'layer_2', 'layer_3']:
                    if f'{layer}_accuracy' not in final_metrics:
                        final_metrics[f'{layer}_accuracy'] = 0.0
                    if f'{layer}_precision' not in final_metrics:
                        final_metrics[f'{layer}_precision'] = 0.0
                    if f'{layer}_recall' not in final_metrics:
                        final_metrics[f'{layer}_recall'] = 0.0
                    if f'{layer}_f1' not in final_metrics:
                        final_metrics[f'{layer}_f1'] = 0.0
                
                # Also ensure validation metrics are included
                for layer in ['layer_1', 'layer_2', 'layer_3']:
                    if f'val_{layer}_accuracy' not in final_metrics:
                        final_metrics[f'val_{layer}_accuracy'] = 0.0
                    if f'val_{layer}_precision' not in final_metrics:
                        final_metrics[f'val_{layer}_precision'] = 0.0
                    if f'val_{layer}_recall' not in final_metrics:
                        final_metrics[f'val_{layer}_recall'] = 0.0
                    if f'val_{layer}_f1' not in final_metrics:
                        final_metrics[f'val_{layer}_f1'] = 0.0

                # Emit metrics callback for UI
                if self.emit_metrics_callback:
                    phase_name = 'training_phase_single' if self._is_single_phase else f'training_phase_{phase_num}'
                    self.emit_metrics_callback(phase_name, epoch + 1, final_metrics)
                
                # Emit live chart data
                self._emit_training_charts(epoch, phase_num, final_metrics, layer_metrics)
                
                # Update visualization manager
                self._update_visualization_manager(epoch, phase_num, final_metrics, layer_metrics)
                
                # Check for best model and save checkpoint
                is_best = metrics_tracker.is_best_model()
                if is_best:
                    best_checkpoint_path = self._save_checkpoint(epoch, final_metrics, phase_num)
                    best_metrics = final_metrics.copy()
                
                # Scheduler step
                self._handle_scheduler_step(scheduler, final_metrics)
                
                # Early stopping check
                should_stop = self._handle_early_stopping(early_stopping, final_metrics, epoch, phase_num)
                if should_stop:
                    # Ensure we have valid results even if early stopped
                    if not best_metrics:
                        best_metrics = final_metrics.copy()
                    if not best_checkpoint_path:
                        best_checkpoint_path = self._save_checkpoint(epoch, final_metrics, phase_num)
                    
                    # Complete epoch tracking due to early stopping
                    self.progress_tracker.complete_epoch_early_stopping(
                        epoch + 1, 
                        f"Early stopping triggered - no improvement for {early_stopping.patience} epochs"
                    )
                    break
                
                # Update epoch progress - normal completion
                epoch_duration = time.time() - epoch_start_time
                self.progress_tracker.update_epoch_progress(
                    epoch + 1, epochs,
                    f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.1f}s - Loss: {final_metrics.get('train_loss', 0):.4f}"
                )
            
            # Ensure we have valid results even if early stopped
            if not best_metrics:
                best_metrics = final_metrics.copy()
            
            if not best_checkpoint_path:
                best_checkpoint_path = self._save_checkpoint(epoch, final_metrics, phase_num)
            
            return {
                'success': True,
                'epochs_completed': epoch + 1,
                'best_metrics': best_metrics,
                'best_checkpoint': best_checkpoint_path,
                'final_metrics': final_metrics,
                'training_completed_successfully': True
            }
            
        except Exception as e:
            logger.error(f"Error in training phase {phase_num}: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _train_epoch(self, train_loader, optimizer, loss_manager, scaler, 
                    epoch: int, total_epochs: int, phase_num: int) -> Dict[str, float]:
        """Train for one epoch with optimized progress tracking."""
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        
        # Store last batch for metrics (memory efficient)
        last_predictions = None
        last_targets = None
        
        # Progress update frequency for performance
        update_freq = max(1, num_batches // 20)  # Update 20 times per epoch max
        logger.info(f"🔍 Training setup: {num_batches} batches, update_freq={update_freq}")
        
        # Start batch tracking
        self.progress_tracker.start_batch_tracking(num_batches)
        
        logger.info(f"🚀 Starting training loop with {num_batches} batches")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx < 3:  # Debug first 3 batches
                logger.info(f"🔍 Processing training batch {batch_idx + 1}/{num_batches}")
            
            # Move data to device efficiently
            device = next(self.model.parameters()).device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=scaler is not None):
                predictions = self.model(images)
                
                # Normalize predictions format based on mode and phase
                if not isinstance(predictions, dict):
                    current_phase = getattr(self.model, 'current_phase', 1)
                    training_mode = self.config.get('training_mode', 'two_phase')
                    layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
                    
                    if batch_idx == 0:  # Only log on first batch
                        logger.info(f"Training - Model current_phase: {current_phase}, training_mode: {training_mode}, layer_mode: {layer_mode}, prediction type: {type(predictions)}")
                    
                    # Determine if we should use multi-layer format
                    use_multi_layer = False
                    if training_mode == 'single_phase' and layer_mode == 'multi':
                        use_multi_layer = True
                    elif training_mode == 'two_phase' and current_phase == 2:
                        use_multi_layer = True
                    
                    if use_multi_layer:
                        # Multi-layer mode: Convert tuple/list predictions to multi-layer dict format
                        if batch_idx == 0:
                            logger.info(f"Multi-layer training: Converting {type(predictions)} to multi-layer dict")
                        
                        if isinstance(predictions, (tuple, list)) and len(predictions) >= 1:
                            # YOLOv5 returns tuple of predictions for different scales
                            # For multi-layer training, we need to split this into layer predictions
                            # Use the first scale prediction and replicate for all layers
                            base_prediction = predictions[0]  # Use first scale
                            
                            predictions = {
                                'layer_1': base_prediction,
                                'layer_2': base_prediction,  # For now, use same prediction
                                'layer_3': base_prediction   # This should be improved later
                            }
                            if batch_idx == 0:
                                logger.info(f"Created multi-layer predictions: {list(predictions.keys())}")
                        else:
                            # Fallback to single layer
                            predictions = {'layer_1': predictions}
                            if batch_idx == 0:
                                logger.warning(f"Fallback to single layer due to unexpected prediction format")
                    else:
                        # Single layer mode: normalize to layer_1 as expected
                        predictions = {'layer_1': predictions}
                
                # Store only last batch for metrics (memory efficient)
                if batch_idx == num_batches - 1:
                    # Process predictions similar to validation for consistency
                    processed_predictions = {}
                    processed_targets = {}
                    
                    for layer_name, layer_preds in predictions.items():
                        # Process YOLO predictions for classification metrics
                        if isinstance(layer_preds, list) and len(layer_preds) > 0:
                            # Extract classification predictions from YOLO output
                            first_scale = layer_preds[0]
                            if isinstance(first_scale, torch.Tensor) and first_scale.numel() > 0:
                                if first_scale.dim() >= 2 and first_scale.shape[-1] > 5:
                                    # Extract class predictions (after bbox and objectness)
                                    class_logits = first_scale[..., 5:]  # Shape: [batch, detections, num_classes]
                                    objectness = first_scale[..., 4:5]   # Shape: [batch, detections, 1]
                                    
                                    # Apply objectness weighting to class probabilities
                                    weighted_class_probs = torch.sigmoid(objectness) * torch.sigmoid(class_logits)
                                    
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
                                            batch_size = weighted_class_probs.shape[0]
                                            # Flatten spatial dimensions for each image and take max
                                            reshaped_probs = weighted_class_probs.view(batch_size, -1, weighted_class_probs.shape[-1])
                                            reshaped_obj = objectness_scores.view(batch_size, -1)
                                            
                                            best_detection_idx = torch.argmax(reshaped_obj, dim=1)  # [batch]
                                            batch_indices = torch.arange(batch_size, device=device)
                                            layer_output = reshaped_probs[batch_indices, best_detection_idx]  # [batch, num_classes]
                                    else:
                                        layer_output = torch.zeros((images.shape[0], 7), device=device)
                                else:
                                    layer_output = torch.zeros((images.shape[0], 7), device=device)
                            else:
                                layer_output = torch.zeros((images.shape[0], 7), device=device)
                        else:
                            # Direct tensor - assume it's already in the right format
                            layer_output = layer_preds if isinstance(layer_preds, torch.Tensor) else torch.zeros((images.shape[0], 7), device=device)
                        
                        processed_predictions[layer_name] = layer_output.detach()
                        
                        # Process targets - extract class labels per image
                        if targets.numel() > 0 and targets.dim() >= 2 and targets.shape[-1] > 1:
                            batch_size = images.shape[0]
                            layer_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
                            
                            for img_idx in range(batch_size):
                                img_targets = targets[targets[:, 0] == img_idx]  # Filter by image index
                                if len(img_targets) > 0:
                                    classes = img_targets[:, 1].long()  # Extract class column
                                    if len(classes) > 0:
                                        layer_targets[img_idx] = classes[0]  # Use first class
                        else:
                            layer_targets = torch.zeros(images.shape[0], dtype=torch.long, device=device)
                        
                        processed_targets[layer_name] = layer_targets.detach()
                    
                    last_predictions = processed_predictions
                    last_targets = processed_targets
                
                # Calculate loss
                loss, _ = loss_manager.compute_loss(predictions, targets, images.shape[-1])
            
            # Backward pass
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            # Efficient progress updates
            if batch_idx % update_freq == 0 or batch_idx == num_batches - 1:
                avg_loss = running_loss / (batch_idx + 1)
                
                if batch_idx < 5 or batch_idx % 10 == 0:  # Debug first few and every 10th batch
                    logger.info(f"🔍 Training batch {batch_idx + 1}/{num_batches}, avg_loss={avg_loss:.4f}")
                
                # Update batch progress
                self.progress_tracker.update_batch_progress(
                    batch_idx + 1, num_batches,
                    f"Training batch {batch_idx + 1}/{num_batches}",
                    loss=avg_loss,
                    epoch=epoch + 1
                )
        
        # Store for final metrics calculation
        if last_predictions and last_targets:
            self._last_predictions = last_predictions
            self._last_targets = last_targets
        
        # Complete batch tracking
        self.progress_tracker.complete_batch_tracking()
        
        return {'train_loss': running_loss / num_batches}
    
    def _validate_epoch(self, val_loader, loss_manager, 
                       epoch: int, total_epochs: int, phase_num: int) -> Dict[str, float]:
        """Optimized validation for one epoch."""
        self.model.eval()
        running_val_loss = 0.0
        num_batches = len(val_loader)
        
        # Debug logging
        logger.info(f"Starting validation epoch {epoch+1} with {num_batches} batches")
        if num_batches == 0:
            logger.warning("Validation loader is empty!")
            return {
                'val_loss': 0.0,
                'val_map50': 0.0,
                'val_map50_95': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_f1': 0.0,
                'val_accuracy': 0.0
            }
        
        # Initialize metrics dictionary and collectors for actual metrics computation
        metrics = {}
        all_predictions = {}  # Collect predictions for classification metrics computation
        all_targets = {}      # Collect targets for classification metrics computation
        
        # Initialize AP calculator for mAP computation
        from smartcash.model.training.metrics_tracker import APCalculator
        ap_calculator = APCalculator()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if batch_idx == 0 or batch_idx % 10 == 0:  # Log first and every 10th batch
                    logger.info(f"Processing validation batch {batch_idx+1}/{num_batches}, images: {images.shape}, targets: {targets.shape}")
                
                device = next(self.model.parameters()).device
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Get model predictions
                predictions = self.model(images)
                if batch_idx == 0:  # Only log on first batch to reduce noise
                    logger.info(f"Model predictions type: {type(predictions)}, structure: {type(predictions).__name__}")
                    if isinstance(predictions, (list, tuple)):
                        logger.info(f"  Predictions list length: {len(predictions)}")
                        if len(predictions) > 0:
                            logger.info(f"  First prediction type: {type(predictions[0])}, shape: {getattr(predictions[0], 'shape', 'N/A')}")
                
                # Handle prediction format based on mode and phase - validation loop
                if not isinstance(predictions, dict):
                    current_phase = getattr(self.model, 'current_phase', 1)
                    training_mode = self.config.get('training_mode', 'two_phase')
                    layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
                    
                    logger.info(f"Validation - Model current_phase: {current_phase}, training_mode: {training_mode}, layer_mode: {layer_mode}, prediction type: {type(predictions)}")
                    
                    # Determine if we should use multi-layer format
                    use_multi_layer = False
                    if training_mode == 'single_phase' and layer_mode == 'multi':
                        use_multi_layer = True
                    elif training_mode == 'two_phase' and current_phase == 2:
                        use_multi_layer = True
                    
                    if use_multi_layer:
                        # Multi-layer mode: Convert tuple/list predictions to multi-layer dict format
                        logger.info(f"Multi-layer validation: Converting {type(predictions)} to multi-layer dict")
                        
                        if isinstance(predictions, (tuple, list)) and len(predictions) >= 1:
                            # YOLOv5 returns tuple of predictions for different scales
                            # For multi-layer training, we need to split this into layer predictions
                            # Use the first scale prediction and replicate for all layers
                            base_prediction = predictions[0]  # Use first scale
                            
                            predictions = {
                                'layer_1': base_prediction,
                                'layer_2': base_prediction,  # For now, use same prediction
                                'layer_3': base_prediction   # This should be improved later
                            }
                            logger.info(f"Created multi-layer predictions: {list(predictions.keys())}")
                        else:
                            # Fallback to single layer
                            predictions = {'layer_1': predictions}
                            logger.warning(f"Fallback to single layer due to unexpected prediction format")
                    else:
                        # Single layer mode: normalize to layer_1 as expected
                        predictions = {'layer_1': predictions}
                
                # Compute loss and metrics
                try:
                    loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
                    running_val_loss += loss.item()
                    
                    if batch_idx == 0:
                        logger.info(f"First batch loss: {loss.item():.4f}, loss_breakdown keys: {list(loss_breakdown.keys())}")
                        
                except Exception as e:
                    logger.error(f"Error computing loss in validation batch {batch_idx}: {e}")
                    continue
                
                # Collect predictions and targets for metrics computation
                try:
                    for layer_name, layer_preds in predictions.items():
                        if layer_name not in all_predictions:
                            all_predictions[layer_name] = []
                            all_targets[layer_name] = []
                        
                        # Process YOLO predictions for both classification metrics and mAP
                        if batch_idx == 0:  # Debug only first batch to reduce noise
                            logger.info(f"mAP Debug - batch {batch_idx}: layer_preds type: {type(layer_preds)}, is_list_or_tuple: {isinstance(layer_preds, (list, tuple))}, len: {len(layer_preds) if isinstance(layer_preds, (list, tuple)) else 'N/A'}")
                            
                        # Handle list, tuple, and tensor formats for mAP computation
                        mAP_processed = False
                        if isinstance(layer_preds, (list, tuple)) and len(layer_preds) > 0:
                            # List/tuple format (expected YOLOv5 multi-scale output)
                            try:
                                if batch_idx == 0:
                                    logger.info(f"mAP Debug - batch {batch_idx}: Processing {type(layer_preds).__name__} format with {len(layer_preds)} scales")
                                self._add_to_map_calculator(ap_calculator, layer_preds, targets, images.shape, device, batch_idx)
                                mAP_processed = True
                            except Exception as e:
                                if batch_idx < 5:
                                    logger.warning(f"mAP processing failed for batch {batch_idx}: {e}")
                                    import traceback
                                    logger.warning(f"mAP traceback: {traceback.format_exc()}")
                                    
                        elif isinstance(layer_preds, torch.Tensor) and layer_preds.numel() > 0:
                            # Single tensor format - wrap in list for compatibility
                            try:
                                if batch_idx == 0:
                                    logger.info(f"mAP Debug - batch {batch_idx}: Processing tensor format {layer_preds.shape}, wrapping in list")
                                wrapped_preds = [layer_preds]
                                self._add_to_map_calculator(ap_calculator, wrapped_preds, targets, images.shape, device, batch_idx)
                                mAP_processed = True
                            except Exception as e:
                                if batch_idx < 5:
                                    logger.warning(f"mAP processing (tensor) failed for batch {batch_idx}: {e}")
                                    import traceback
                                    logger.warning(f"mAP traceback: {traceback.format_exc()}")
                                    
                        if batch_idx == 0 and not mAP_processed:
                            logger.warning(f"mAP Debug - batch {batch_idx}: No mAP processing - layer_preds type: {type(layer_preds)}, shape/len: {getattr(layer_preds, 'shape', len(layer_preds) if hasattr(layer_preds, '__len__') else 'N/A')}")
                            
                            # Extract classification predictions from YOLO output
                            # YOLO format: [batch, anchors*grid_cells, 5+num_classes]
                            # where 5 = [x, y, w, h, objectness] and then class probabilities
                            first_scale = layer_preds[0]
                            if isinstance(first_scale, torch.Tensor) and first_scale.numel() > 0:
                                if first_scale.dim() >= 2 and first_scale.shape[-1] > 5:
                                    # Extract class predictions (after bbox and objectness)
                                    class_logits = first_scale[..., 5:]  # Shape: [batch, detections, num_classes]
                                    objectness = first_scale[..., 4:5]   # Shape: [batch, detections, 1]
                                    
                                    # Apply objectness weighting to class probabilities
                                    weighted_class_probs = torch.sigmoid(objectness) * torch.sigmoid(class_logits)
                                    
                                    # For accuracy calculation, we need to get the best prediction per image
                                    # Method: Take the detection with highest objectness score per image
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
                                            batch_size = weighted_class_probs.shape[0]
                                            # Flatten spatial dimensions for each image and take max
                                            reshaped_probs = weighted_class_probs.view(batch_size, -1, weighted_class_probs.shape[-1])
                                            reshaped_obj = objectness_scores.view(batch_size, -1)
                                            
                                            best_detection_idx = torch.argmax(reshaped_obj, dim=1)  # [batch]
                                            batch_indices = torch.arange(batch_size, device=device)
                                            layer_output = reshaped_probs[batch_indices, best_detection_idx]  # [batch, num_classes]
                                    else:
                                        # No detections, use zero predictions
                                        batch_size = images.shape[0]
                                        layer_output = torch.zeros((batch_size, 7), device=device)
                                else:
                                    batch_size = images.shape[0]
                                    layer_output = torch.zeros((batch_size, 7), device=device)
                            else:
                                layer_output = torch.zeros((images.shape[0], 7), device=device)
                        else:
                            # Direct tensor - assume it's already in the right format
                            layer_output = layer_preds if isinstance(layer_preds, torch.Tensor) else torch.zeros((images.shape[0], 7), device=device)
                        
                        all_predictions[layer_name].append(layer_output)
                        
                        # Extract target classes for this layer
                        # For YOLO training, targets format: [image_idx, class_id, x_center, y_center, width, height]
                        if targets.numel() > 0 and targets.dim() >= 2 and targets.shape[-1] > 1:
                            # Get unique images in this batch
                            batch_size = images.shape[0]
                            layer_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
                            
                            # For each image in the batch, find the most common class (or first class if tie)
                            for img_idx in range(batch_size):
                                # Find all targets for this image
                                img_targets = targets[targets[:, 0] == img_idx]  # Filter by image index
                                if len(img_targets) > 0:
                                    # Get the most frequent class, or first class if multiple objects
                                    classes = img_targets[:, 1].long()  # Extract class column
                                    if len(classes) > 0:
                                        # Use the first valid class (simple approach for multi-object images)
                                        layer_targets[img_idx] = classes[0]
                                # If no targets for this image, it remains 0 (background/negative)
                        else:
                            layer_targets = torch.zeros(images.shape[0], dtype=torch.long, device=device)
                        all_targets[layer_name].append(layer_targets)
                        
                        if batch_idx == 0:
                            logger.info(f"Layer {layer_name}: prediction shape {layer_output.shape}, target shape {layer_targets.shape}")
                            
                except Exception as e:
                    logger.error(f"Error processing predictions in batch {batch_idx}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Update metrics from loss breakdown (these will be mostly zeros for validation metrics)
                for k, v in loss_breakdown.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    if k.startswith('val_') or k in ['map50', 'map50_95', 'precision', 'recall', 'f1', 'accuracy']:
                        if k not in metrics:
                            metrics[k] = 0.0
                        metrics[k] += v
                
                # Update progress
                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    avg_loss = running_val_loss / (batch_idx + 1)
                    self.progress_tracker.update_phase(
                        epoch, total_epochs,
                        f"Val Batch {batch_idx + 1}/{num_batches}",
                        epoch=epoch + 1,
                        batch_idx=batch_idx + 1,
                        batch_total=num_batches,
                        metrics={'val_loss': avg_loss, **{k: v / (batch_idx + 1) for k, v in metrics.items()}}
                    )
        
        # Debug logging for validation results
        logger.info(f"Validation completed: {num_batches} batches processed, running_val_loss: {running_val_loss}")
        logger.info(f"Collected predictions: {list(all_predictions.keys())}")
        logger.info(f"Collected targets: {list(all_targets.keys())}")
        
        # Calculate average metrics from loss breakdown
        if num_batches > 0:
            metrics = {k: v / num_batches for k, v in metrics.items()}
        
        # Compute mAP metrics using AP calculator
        map_metrics = {}
        try:
            # Check if AP calculator has any data
            num_predictions = len(ap_calculator.predictions)
            num_targets = len(ap_calculator.targets)
            logger.info(f"mAP Calculator Data: {num_predictions} predictions, {num_targets} targets")
            
            if num_predictions == 0 or num_targets == 0:
                logger.warning(f"⚠️ Insufficient data for mAP computation: {num_predictions} predictions, {num_targets} targets")
                map_metrics['val_map50'] = 0.0
                map_metrics['val_map50_95'] = 0.0
            else:
                # Compute mAP@0.5
                map50, class_aps = ap_calculator.compute_map(iou_threshold=0.5)
                map_metrics['val_map50'] = float(map50)
                
                # Compute mAP@0.5:0.95
                map50_95 = ap_calculator.compute_map50_95()
                map_metrics['val_map50_95'] = float(map50_95)
                
                logger.info(f"✅ Computed mAP metrics: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}")
                if class_aps:
                    logger.debug(f"Per-class APs: {class_aps}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error computing mAP (falling back to 0.0): {e}")
            import traceback
            logger.debug(f"mAP computation traceback: {traceback.format_exc()}")
            map_metrics['val_map50'] = 0.0
            map_metrics['val_map50_95'] = 0.0
        
        # Compute actual validation metrics from collected predictions and targets
        computed_metrics = {}
        if all_predictions and all_targets:
            # Concatenate all predictions and targets
            final_predictions = {}
            final_targets = {}
            
            for layer_name in all_predictions.keys():
                if all_predictions[layer_name] and all_targets[layer_name]:
                    try:
                        final_predictions[layer_name] = torch.cat(all_predictions[layer_name], dim=0)
                        final_targets[layer_name] = torch.cat(all_targets[layer_name], dim=0)
                    except Exception as e:
                        logger.debug(f"Error concatenating {layer_name} data: {e}")
                        continue
            
            # Calculate metrics using the metrics utils
            if final_predictions and final_targets:
                computed_metrics = calculate_multilayer_metrics(final_predictions, final_targets)
                logger.info(f"Computed validation metrics: {computed_metrics}")
        
        # Ensure all required metrics are present
        base_metrics = {
            'val_loss': running_val_loss / num_batches if num_batches > 0 else 0.0,
            'val_map50': map_metrics.get('val_map50', 0.0),
            'val_map50_95': map_metrics.get('val_map50_95', 0.0),
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0
        }
        
        # Update with calculated metrics from loss breakdown (but preserve val_loss and mAP)
        correct_val_loss = base_metrics['val_loss']  # Save correct val_loss
        correct_map50 = base_metrics['val_map50']    # Save correct mAP@0.5  
        correct_map50_95 = base_metrics['val_map50_95']  # Save correct mAP@0.5:0.95
        
        base_metrics.update(metrics)
        
        # Restore correct values that shouldn't be overwritten
        base_metrics['val_loss'] = correct_val_loss
        base_metrics['val_map50'] = correct_map50        # Restore correct mAP@0.5
        base_metrics['val_map50_95'] = correct_map50_95  # Restore correct mAP@0.5:0.95
        
        # Update with computed metrics (precision, recall, F1, accuracy)
        if computed_metrics:
            # For multi-layer models, average the metrics across layers
            layer_count = 0
            total_accuracy = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            
            for key, value in computed_metrics.items():
                if 'accuracy' in key:
                    total_accuracy += value
                    layer_count += 1
                elif 'precision' in key:
                    total_precision += value
                elif 'recall' in key:
                    total_recall += value
                elif 'f1' in key:
                    total_f1 += value
            
            # Update validation metrics with averages
            if layer_count > 0:
                base_metrics['val_accuracy'] = total_accuracy / layer_count
                base_metrics['val_precision'] = total_precision / layer_count
                base_metrics['val_recall'] = total_recall / layer_count
                base_metrics['val_f1'] = total_f1 / layer_count
            
            # Also include individual layer metrics in the output
            base_metrics.update(computed_metrics)
        
        return base_metrics
    
    def _add_to_map_calculator(self, ap_calculator, layer_preds, targets, image_shape, device, batch_idx):
        """
        Extract bounding box predictions and add to mAP calculator
        
        Args:
            ap_calculator: APCalculator instance
            layer_preds: List of YOLO predictions [scales]
            targets: Target tensor [num_targets, 6] format [batch_idx, class, x, y, w, h]
            image_shape: Shape of input images
            device: Device for tensors
            batch_idx: Current batch index for logging
        """
        try:
            # Process first scale for simplicity (can be extended to all scales)
            scale_pred = layer_preds[0]
            if batch_idx < 2:  # Log format for first few batches
                logger.info(f"mAP Debug - batch {batch_idx}: layer_preds type {type(layer_preds)}, length {len(layer_preds) if hasattr(layer_preds, '__len__') else 'N/A'}")
                logger.info(f"mAP Debug - batch {batch_idx}: scale_pred type {type(scale_pred)}, shape {scale_pred.shape if hasattr(scale_pred, 'shape') else 'N/A'}")
            
            # Handle both formats: 4D [batch, anchors, grid_y, grid_x, features] and 3D [batch, detections, features]
            if isinstance(scale_pred, torch.Tensor):
                if scale_pred.dim() == 4:
                    # 4D format [batch, anchors, grid_y, grid_x, features]
                    batch_size, num_anchors, grid_h, grid_w, num_features = scale_pred.shape
                    if num_features < 7:  # Need at least x,y,w,h,obj + 2 classes
                        if batch_idx < 2:
                            logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                        return
                    # Flatten spatial dimensions: [batch, num_anchors * grid_h * grid_w, features]
                    flat_pred = scale_pred.view(batch_size, -1, num_features)
                    
                elif scale_pred.dim() == 3:
                    # 3D format [batch, detections, features] - modern YOLOv5 output
                    batch_size, num_detections, num_features = scale_pred.shape
                    if num_features < 7:  # Need at least x,y,w,h,obj + 2 classes
                        if batch_idx < 2:
                            logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                        return
                    flat_pred = scale_pred
                    if batch_idx < 2:
                        logger.info(f"mAP Debug - batch {batch_idx}: Using 3D format [batch={batch_size}, detections={num_detections}, features={num_features}]")
                else:
                    if batch_idx < 2:
                        logger.warning(f"mAP Debug - batch {batch_idx}: Invalid tensor format - expected 3D or 4D tensor, got {scale_pred.shape}")
                    return
            else:
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: scale_pred is not a tensor: {type(scale_pred)}")
                return
            
            # Extract YOLO prediction components
            obj_conf = torch.sigmoid(flat_pred[..., 4])  # Objectness confidence
            class_logits = flat_pred[..., 5:]       # Class logits
            class_probs = torch.sigmoid(class_logits)  # Class probabilities
            
            # Handle coordinate processing based on format
            img_h, img_w = image_shape[-2:]
            
            if scale_pred.dim() == 4:
                # 4D format: need to convert grid coordinates
                xy = torch.sigmoid(flat_pred[..., :2])  # Center coordinates (0-1)
                wh = torch.exp(flat_pred[..., 2:4])     # Width/height (relative to anchors)
                
                grid_scale_x = img_w / grid_w
                grid_scale_y = img_h / grid_h
                
                # Create grid coordinates
                grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w)
                grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view(1, 1, grid_h, grid_w)
                grid_xy = torch.cat([grid_x, grid_y], dim=1).float()
                grid_xy = grid_xy.view(1, 2, -1).permute(0, 2, 1)  # [1, grid_cells, 2]
                
                # Convert center coordinates to absolute
                xy_abs = (xy + grid_xy) * torch.tensor([grid_scale_x, grid_scale_y], device=device)
                
                # Convert to corner coordinates [x1, y1, x2, y2]
                wh_abs = wh * torch.tensor([grid_scale_x, grid_scale_y], device=device) * 0.5  # Scale factor
                x1 = xy_abs[..., 0] - wh_abs[..., 0]
                y1 = xy_abs[..., 1] - wh_abs[..., 1]
                x2 = xy_abs[..., 0] + wh_abs[..., 0]
                y2 = xy_abs[..., 1] + wh_abs[..., 1]
                
            else:
                # 3D format: coordinates are already processed by YOLOv5
                # Assume format: [x_center, y_center, width, height] in image coordinates
                xy_center = flat_pred[..., :2]  # Center coordinates (already in pixels)
                wh_size = flat_pred[..., 2:4]   # Width/height (already in pixels)
                
                # Convert to corner coordinates [x1, y1, x2, y2] 
                x1 = xy_center[..., 0] - wh_size[..., 0] * 0.5
                y1 = xy_center[..., 1] - wh_size[..., 1] * 0.5
                x2 = xy_center[..., 0] + wh_size[..., 0] * 0.5
                y2 = xy_center[..., 1] + wh_size[..., 1] * 0.5
            
            # Filter by confidence threshold (reasonable threshold to avoid performance issues)
            conf_thresh = 0.1  # Balanced threshold to avoid processing too many predictions
            max_preds_per_img = 1000  # Limit predictions per image for performance
            final_scores = obj_conf * class_probs.max(dim=-1)[0]
            conf_mask = final_scores > conf_thresh
            
            if batch_idx < 2:
                total_detections = conf_mask.sum().item()
                max_obj_conf = obj_conf.max().item() if obj_conf.numel() > 0 else 0
                max_class_prob = class_probs.max().item() if class_probs.numel() > 0 else 0
                max_final_score = final_scores.max().item() if final_scores.numel() > 0 else 0
                logger.info(f"mAP Debug - batch {batch_idx}: Found {total_detections} detections above threshold {conf_thresh}")
                logger.info(f"  Max objectness: {max_obj_conf:.4f}, Max class prob: {max_class_prob:.4f}, Max final score: {max_final_score:.4f}")
                logger.info(f"  Prediction tensor shape: {flat_pred.shape}, Confidence threshold: {conf_thresh}")
                if total_detections == 0:
                    logger.warning(f"  No detections found! All {final_scores.numel()} predictions below threshold")
            
            # Process each image in batch
            for img_idx in range(batch_size):
                img_mask = conf_mask[img_idx]
                if not img_mask.any():
                    continue
                
                # Limit number of predictions per image for performance
                img_scores = final_scores[img_idx][img_mask]
                if len(img_scores) > max_preds_per_img:
                    # Keep only top-scored predictions
                    top_k_indices = torch.topk(img_scores, max_preds_per_img)[1]
                    img_indices = torch.where(img_mask)[0][top_k_indices]
                    img_mask = torch.zeros_like(img_mask)
                    img_mask[img_indices] = True
                
                # Get valid predictions for this image
                valid_boxes = torch.stack([
                    x1[img_idx][img_mask],
                    y1[img_idx][img_mask], 
                    x2[img_idx][img_mask],
                    y2[img_idx][img_mask]
                ], dim=-1)
                
                valid_scores = final_scores[img_idx][img_mask]
                valid_classes = class_probs[img_idx][img_mask].argmax(dim=-1)
                
                # Get ground truth for this image
                img_targets = targets[targets[:, 0] == img_idx]  # Filter by batch index
                if len(img_targets) == 0:
                    continue
                
                # Convert target format [class, x_center, y_center, width, height] to [x1, y1, x2, y2]
                gt_classes = img_targets[:, 1].long()
                gt_centers = img_targets[:, 2:4] * torch.tensor([img_w, img_h], device=device)
                gt_sizes = img_targets[:, 4:6] * torch.tensor([img_w, img_h], device=device)
                gt_x1 = gt_centers[:, 0] - gt_sizes[:, 0] * 0.5
                gt_y1 = gt_centers[:, 1] - gt_sizes[:, 1] * 0.5
                gt_x2 = gt_centers[:, 0] + gt_sizes[:, 0] * 0.5
                gt_y2 = gt_centers[:, 1] + gt_sizes[:, 1] * 0.5
                gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)
                
                # Add to AP calculator (reshape to expected format)
                if len(valid_boxes) > 0 and len(gt_boxes) > 0:
                    if batch_idx < 2:
                        logger.info(f"mAP Debug - batch {batch_idx}, img {img_idx}: Adding {len(valid_boxes)} predictions (max {max_preds_per_img}) and {len(gt_boxes)} targets to AP calculator")
                        logger.info(f"  Pred scores range: {valid_scores.min():.4f} - {valid_scores.max():.4f}")
                        logger.info(f"  Pred classes: {valid_classes.unique().tolist()}")
                        logger.info(f"  GT classes: {gt_classes.unique().tolist()}")
                    
                    ap_calculator.add_batch(
                        pred_boxes=valid_boxes.unsqueeze(0),      # [1, num_preds, 4]
                        pred_scores=valid_scores.unsqueeze(0),    # [1, num_preds]
                        pred_classes=valid_classes.unsqueeze(0),  # [1, num_preds]
                        true_boxes=gt_boxes.unsqueeze(0),         # [1, num_targets, 4]
                        true_classes=gt_classes.unsqueeze(0),     # [1, num_targets]
                        image_ids=[img_idx]                       # [1]
                    )
                elif batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}, img {img_idx}: Skipping - valid_boxes: {len(valid_boxes)}, gt_boxes: {len(gt_boxes)}")
                    
        except Exception as e:
            if batch_idx < 3:  # Only log first few failures
                logger.warning(f"mAP extraction failed for batch {batch_idx}: {e}")
                import traceback
                logger.warning(f"mAP extraction traceback: {traceback.format_exc()}")
    
    def _emit_training_charts(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """Emit live chart data for training visualization."""
        if not self.emit_live_chart_callback:
            return
            
        # Determine active layers using the same logic as the metrics callback
        show_layers = self._determine_active_layers_for_charts(phase_num, final_metrics)
        
        # Training curves chart - use primary active layer for accuracy
        primary_layer = show_layers[0] if show_layers else 'layer_1'
        chart_data = {
            'epoch': epoch + 1,
            'train_loss': final_metrics.get('train_loss', 0),
            'val_loss': final_metrics.get('val_loss', 0),
            'train_accuracy': final_metrics.get(f'{primary_layer}_accuracy', 0),
            'val_accuracy': final_metrics.get('val_accuracy', 0),  # Use global val_accuracy, not layer-specific
            'phase': phase_num
        }
        # Fix nested f-string syntax error
        phase_label = 'Single Phase' if self._is_single_phase else f'Phase {phase_num}'
        self.emit_live_chart_callback('training_curves', chart_data, {
            'title': f'Training Progress - {phase_label}',
            'xlabel': 'Epoch',
            'ylabel': 'Loss / Accuracy'
        })
        
        # Layer metrics chart - only include active layers
        layer_chart_data = {}
        for layer in show_layers:
            layer_chart_data[layer] = {
                'accuracy': final_metrics.get(f'{layer}_accuracy', 0),
                'precision': final_metrics.get(f'{layer}_precision', 0),
                'recall': final_metrics.get(f'{layer}_recall', 0),
                'f1': final_metrics.get(f'{layer}_f1', 0)
            }
        
        # Only emit chart if we have meaningful data for at least one layer
        if layer_chart_data and any(layer_chart_data[layer]['accuracy'] > 0 for layer in layer_chart_data):
            self.emit_live_chart_callback('layer_metrics', {
                'epoch': epoch + 1,
                'layers': layer_chart_data,
                'phase': phase_num
            }, {
                'title': f'Layer Performance - {"Single Phase" if self._is_single_phase else "Phase " + str(phase_num)}',
                'xlabel': 'Layer',
                'ylabel': 'Metric Value'
            })
    
    def _determine_active_layers_for_charts(self, phase_num: int, final_metrics: dict) -> list:
        """Determine which layers should be included in charts using the same logic as metrics callback."""
        if self._is_single_phase:
            # Single-phase mode: Determine active layers from actual metrics
            layer_activity = {}
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                # Check if this layer has any meaningful metrics
                has_activity = any(
                    final_metrics.get(f'{layer}_{metric}', 0) > 0.0001 or 
                    final_metrics.get(f'val_{layer}_{metric}', 0) > 0.0001
                    for metric in ['accuracy', 'precision', 'recall', 'f1']
                )
                layer_activity[layer] = has_activity
            
            # Determine active layers
            active_layers = [layer for layer, active in layer_activity.items() if active]
            if len(active_layers) == 1:
                # Single-phase, single-layer mode: only show the active layer
                return active_layers
            else:
                # Single-phase, multi-layer mode: show all layers
                return ['layer_1', 'layer_2', 'layer_3']
        else:
            # Two-phase mode
            if phase_num == 1:
                # Phase 1: Only show layer_1
                return ['layer_1']
            else:
                # Phase 2: Show all layers
                return ['layer_1', 'layer_2', 'layer_3']
    
    def _update_visualization_manager(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """Update visualization manager with current epoch data."""
        if not self.visualization_manager:
            return
            
        # Determine active layers using the same logic as charts
        show_layers = self._determine_active_layers_for_charts(phase_num, final_metrics)
        
        phase_name = f"phase_{phase_num}"
        # Create simulated predictions and targets for visualization - only for active layers
        viz_predictions = {}
        viz_targets = {}
        if layer_metrics:
            np.random.seed(epoch * 42)
            for layer in show_layers:
                # Use consistent class numbers based on layer
                num_classes = {'layer_1': 7, 'layer_2': 7, 'layer_3': 3}.get(layer, 7)
                viz_predictions[layer] = np.random.rand(50, num_classes)
                viz_targets[layer] = np.eye(num_classes)[np.random.randint(0, num_classes, 50)]
        
        self.visualization_manager.update_metrics(
            epoch=epoch + 1,
            phase=phase_name,
            metrics=final_metrics,
            predictions=viz_predictions,
            ground_truth=viz_targets
        )
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], phase_num: int) -> str:
        """Save checkpoint using model API or direct save."""
        try:
            checkpoint_dir = Path(self.config['paths']['checkpoints'])
            backbone = self.config['model']['backbone']
            
            # Try model API first
            if self.model_api:
                checkpoint_info = {
                    'epoch': epoch,
                    'phase': phase_num,
                    'metrics': metrics,
                    'is_best': True,
                    'config': self.config
                }
                
                saved_path = self.model_api.save_checkpoint(**checkpoint_info)
                if saved_path:
                    device = next(self.model.parameters()).device
                    device_type = 'cpu' if device.type == 'cpu' else ('mps' if device.type == 'mps' else 'gpu')
                    layer_mode = self.config['model'].get('layer_mode', 'multi')
                    logger.info(f"💾 Best checkpoint saved: {Path(saved_path).name}")
                    logger.info(f"   Epoch: {epoch + 1}, Phase: {phase_num}, Device: {device_type}, Layer mode: {layer_mode}")
                    return saved_path
            
            # Fallback: direct save using utils
            checkpoint_name = generate_checkpoint_name(backbone, self.model, self.config, is_best=True)
            checkpoint_path = checkpoint_dir / checkpoint_name
            
            success = save_checkpoint_to_disk(
                checkpoint_path=checkpoint_path,
                model_state_dict=self.model.state_dict(),
                epoch=epoch,
                phase=phase_num,
                metrics=metrics,
                config=self.config,
                session_id=getattr(self, 'training_session_id', None)
            )
            
            if success:
                device = next(self.model.parameters()).device
                device_type = 'cpu' if device.type == 'cpu' else ('mps' if device.type == 'mps' else 'gpu')
                layer_mode = self.config['model'].get('layer_mode', 'multi')
                logger.info(f"   Epoch: {epoch + 1}, Phase: {phase_num}, Device: {device_type}, Layer mode: {layer_mode}")
                return str(checkpoint_path)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return None
    
    def _handle_scheduler_step(self, scheduler, final_metrics: dict):
        """Handle learning rate scheduler step."""
        if scheduler:
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    monitor_metric = final_metrics.get('val_map50', 0)
                    scheduler.step(monitor_metric)
                else:
                    scheduler.step()
    
    def _handle_early_stopping(self, early_stopping, final_metrics: dict, epoch: int, phase_num: int) -> bool:
        """Handle early stopping logic."""
        monitor_metric = final_metrics.get('val_map50', 0)
        should_stop = early_stopping(monitor_metric, self.model, epoch)
        
        if should_stop:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            logger.info(f"   Best {early_stopping.metric}: {early_stopping.best_score:.6f} at epoch {early_stopping.best_epoch + 1}")
            
            # Add early stopping info to final metrics
            final_metrics['early_stopped'] = True
            final_metrics['early_stop_epoch'] = epoch + 1
            final_metrics['early_stop_reason'] = f"No improvement in {early_stopping.metric} for {early_stopping.patience} epochs"
        
        return should_stop
    
    def set_single_phase_mode(self, is_single_phase: bool):
        """Set single phase mode flag for proper logging."""
        self._is_single_phase = is_single_phase
    
    def _set_model_phase(self, phase_num: int):
        """Set current phase on model for layer mode control, handling nested YOLOv5 structure."""
        phase_set = False
        
        # Force create and set current_phase on the main model
        self.model.current_phase = phase_num
        phase_set = True
        logger.info(f"🎯 Set model.current_phase to {phase_num}")
        
        # Try to set phase on YOLOv5 model if it exists
        if hasattr(self.model, 'yolov5_model'):
            yolo_model = self.model.yolov5_model
            yolo_model.current_phase = phase_num
            phase_set = True
            logger.info(f"🎯 Set yolov5_model.current_phase to {phase_num}")
        
        # Try to set phase on nested model if it exists  
        if hasattr(self.model, 'model'):
            nested_model = self.model.model
            nested_model.current_phase = phase_num
            phase_set = True
            logger.info(f"🎯 Set model.model.current_phase to {phase_num}")
            
            # Check for detection head in nested model
            if hasattr(nested_model, 'model') and hasattr(nested_model.model, '__iter__'):
                try:
                    # Look for detection head (usually the last layer)
                    if len(nested_model.model) > 0:
                        last_layer = nested_model.model[-1]
                        last_layer.current_phase = phase_num
                        logger.info(f"🎯 Set detection_head.current_phase to {phase_num}")
                except Exception as e:
                    logger.debug(f"Could not set phase on detection head: {e}")
        
        # Try to set phase on deeply nested YOLOv5 model
        if hasattr(self.model, 'yolov5_model') and hasattr(self.model.yolov5_model, 'model'):
            deep_model = self.model.yolov5_model.model
            deep_model.current_phase = phase_num
            logger.info(f"🎯 Set yolov5_model.model.current_phase to {phase_num}")
        
        # Also check for detection head specifically (where the phase might be needed)
        if hasattr(self.model, 'head'):
            self.model.head.current_phase = phase_num
            logger.info(f"🎯 Set model.head.current_phase to {phase_num}")
        
        # Check nested head in YOLOv5 model
        if hasattr(self.model, 'yolov5_model') and hasattr(self.model.yolov5_model, 'head'):
            yolo_head = self.model.yolov5_model.head
            yolo_head.current_phase = phase_num
            logger.info(f"🎯 Set yolov5_model.head.current_phase to {phase_num}")
        
        if phase_set:
            logger.info(f"✅ Model phase successfully set to {phase_num} for layer mode control")
        else:
            logger.warning(f"⚠️ Could not find current_phase attribute to set phase {phase_num}")