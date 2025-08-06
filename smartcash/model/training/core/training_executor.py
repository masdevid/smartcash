#!/usr/bin/env python3
"""
Training execution for the unified training pipeline.

This module handles the actual training loop execution including
batch processing, forward/backward passes, and progress tracking.
"""

import torch
from torch.amp import autocast
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from .prediction_processor import PredictionProcessor

logger = get_logger(__name__)


class TrainingExecutor:
    """Handles training epoch execution and batch processing."""
    
    def __init__(self, model, config, progress_tracker):
        """
        Initialize training executor.
        
        Args:
            model: PyTorch model
            config: Training configuration
            progress_tracker: Progress tracking instance
        """
        self.model = model
        self.config = config
        self.progress_tracker = progress_tracker
        self.prediction_processor = PredictionProcessor(config, model)
        
        # State for metrics calculation - accumulate all batches
        self._accumulated_predictions = {}
        self._accumulated_targets = {}
        self._batch_count = 0
    
    def train_epoch(self, train_loader, optimizer, loss_manager, scaler, 
                   epoch: int, total_epochs: int, phase_num: int, display_epoch: int = None) -> Dict[str, float]:
        """
        Train for one epoch with optimized progress tracking.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
            loss_manager: Loss manager instance
            scaler: Mixed precision scaler
            epoch: Current epoch number (0-based)
            total_epochs: Total number of epochs
            phase_num: Current phase number
            display_epoch: Display epoch number (1-based, for progress/logging)
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        
        # Reset accumulated metrics for new epoch
        self._reset_accumulated_metrics()
        
        # Progress update frequency optimized for maximum performance
        update_freq = max(1, num_batches // 20)  # Update 20 times per epoch max for better performance
        
        # Calculate display epoch if not provided
        if display_epoch is None:
            display_epoch = epoch + 1
        
        # Start batch tracking
        self.progress_tracker.start_batch_tracking(num_batches)
        
        # Log batch size information only once (cached)
        if epoch == 0 and not hasattr(self, '_batch_info_logged'):
            logger.info(f"📊 Training Batch Configuration:")
            logger.info(f"   • Configured Batch Size: {train_loader.batch_size}")
            logger.info(f"   • Total Batches: {num_batches}")
            logger.info(f"   • Samples per Epoch: ~{num_batches * train_loader.batch_size}")
            self._batch_info_logged = True
        
        logger.info(f"🚀 Starting training epoch {display_epoch}/{total_epochs} with {num_batches} batches")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Check for shutdown signal much less frequently for better performance
            if batch_idx % 50 == 0:  # Check every 50 batches for maximum performance
                from smartcash.model.training.utils.signal_handler import is_shutdown_requested
                if is_shutdown_requested():
                    logger.info("🛑 Shutdown requested during training batch processing")
                    break
            
            # Process batch
            loss, loss_breakdown, predictions, processed_predictions, processed_targets = self._process_training_batch(
                images, targets, loss_manager, batch_idx, num_batches, phase_num
            )
            
            # Store the loss breakdown from the last batch for metrics reporting
            if loss_breakdown:
                self._last_loss_breakdown = loss_breakdown
            
            # Accumulate predictions and targets from all batches for layer metrics
            if processed_predictions and processed_targets:
                self._accumulate_batch_metrics(processed_predictions, processed_targets, batch_idx)
            
            # Backward pass
            self._backward_pass(loss, optimizer, scaler)
            
            running_loss += loss.item()
            
            # Efficient progress updates
            if batch_idx % update_freq == 0 or batch_idx == num_batches - 1:
                avg_loss = running_loss / (batch_idx + 1)
                
                # Update batch progress
                self.progress_tracker.update_batch_progress(
                    batch_idx + 1, num_batches,
                    f"Training batch {batch_idx + 1}/{num_batches}",
                    loss=avg_loss,
                    epoch=display_epoch
                )
        
        # Log final accumulated metrics summary
        if self._batch_count > 0:
            logger.info(f"📊 Training epoch completed: accumulated metrics from {self._batch_count} batches")
            if isinstance(self._accumulated_predictions, dict):
                for layer_name in self._accumulated_predictions.keys():
                    pred_count = len(self._accumulated_predictions[layer_name]) if self._accumulated_predictions[layer_name] is not None else 0
                    target_count = len(self._accumulated_targets.get(layer_name, [])) if self._accumulated_targets.get(layer_name) is not None else 0
                    logger.info(f"    • {layer_name}: {pred_count} predictions, {target_count} targets")
        
        # Complete batch tracking
        self.progress_tracker.complete_batch_tracking()
        
        # Ensure we have a loss breakdown (fallback to empty dict if none stored)
        if not hasattr(self, '_last_loss_breakdown'):
            self._last_loss_breakdown = {}
        
        # Prepare training metrics with accumulated data for layer metrics
        train_metrics = {
            'train_loss': running_loss / num_batches,
            '_accumulated_predictions': self._accumulated_predictions if self._batch_count > 0 else {},
            '_accumulated_targets': self._accumulated_targets if self._batch_count > 0 else {}
        }
        
        return train_metrics
    
    def _process_training_batch(self, images, targets, loss_manager, batch_idx, num_batches, phase_num):
        """Process a single training batch."""
        # Cache device detection for performance
        if not hasattr(self, '_cached_device'):
            self._cached_device = next(self.model.parameters()).device
        
        # Move data to device efficiently
        images = images.to(self._cached_device, non_blocking=True)
        targets = targets.to(self._cached_device, non_blocking=True)
        
        # Forward pass with mixed precision (cache scaler check)
        if not hasattr(self, '_use_amp'):
            self._use_amp = hasattr(self, 'scaler') and self.scaler is not None
        
        with autocast('cuda', enabled=self._use_amp):
            predictions = self.model(images)
            
            # Normalize predictions format
            predictions = self.prediction_processor.normalize_training_predictions(
                predictions, phase_num, batch_idx
            )
            
            # Process predictions for metrics from all batches (sample to manage memory)
            processed_predictions = None
            processed_targets = None
            if self._should_process_batch_for_metrics(batch_idx, num_batches):
                # Choose processing method based on phase and predictions format
                if self._should_use_full_metrics_processing(predictions, phase_num):
                    # Use full processing for Phase 2 multi-layer metrics
                    processed_predictions, processed_targets = self.prediction_processor.process_for_metrics(
                        predictions, targets, images, self._cached_device
                    )
                else:
                    # Use lightweight processing for Phase 1 or single-layer training
                    processed_predictions, processed_targets = self.prediction_processor.process_for_metrics_lightweight(
                        predictions, targets, self._cached_device
                    )
            
            # Calculate loss
            loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
        
        return loss, loss_breakdown, predictions, processed_predictions, processed_targets
    
    def _should_process_batch_for_metrics(self, batch_idx: int, num_batches: int) -> bool:
        """
        Determine if this batch should be processed for metrics (BALANCED OPTIMIZATION).
        
        Process enough batches to get accurate layer metrics while maintaining performance.
        Balance between speed and metrics accuracy.
        
        Args:
            batch_idx: Current batch index
            num_batches: Total number of batches in epoch
            
        Returns:
            True if this batch should be processed for metrics
        """
        # Process every 5th batch for better layer metrics (20% sampling)
        # This provides better accuracy for layer metrics while still optimizing speed
        return batch_idx % 5 == 0 or batch_idx == num_batches - 1  # Always process last batch
    
    def _should_use_full_metrics_processing(self, predictions: Dict, phase_num: int) -> bool:
        """
        Determine if full metrics processing should be used based on phase and prediction format.
        
        Args:
            predictions: Model predictions
            phase_num: Current training phase
            
        Returns:
            True if full processing should be used, False for lightweight processing
        """
        # Use full processing for Phase 2 multi-layer predictions
        if phase_num == 2 and isinstance(predictions, dict):
            # Check if we have multiple layers in predictions
            layer_count = len([k for k in predictions.keys() if k.startswith('layer_')])
            if layer_count > 1:
                return True
        
        # Use full processing if we have multi-layer predictions regardless of phase
        if isinstance(predictions, dict) and len(predictions) > 1:
            return True
        
        # Default to lightweight processing for Phase 1 or single-layer
        return False
    
    def _backward_pass(self, loss, optimizer, scaler):
        """Perform backward pass with optional mixed precision."""
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    def _reset_accumulated_metrics(self):
        """Reset accumulated metrics for a new epoch."""
        self._accumulated_predictions = {}
        self._accumulated_targets = {}
        self._batch_count = 0
    
    def _accumulate_batch_metrics(self, processed_predictions, processed_targets, batch_idx: int):
        """
        Accumulate predictions and targets from current batch.
        
        Args:
            processed_predictions: Processed predictions from current batch
            processed_targets: Processed targets from current batch 
            batch_idx: Current batch index
        """
        try:
            self._batch_count += 1
            
            # Handle dict-based predictions (multi-layer)
            if isinstance(processed_predictions, dict) and isinstance(processed_targets, dict):
                for layer_name in processed_predictions.keys():
                    if layer_name in processed_targets:
                        # Initialize layer if not exists
                        if layer_name not in self._accumulated_predictions:
                            self._accumulated_predictions[layer_name] = []
                            self._accumulated_targets[layer_name] = []
                        
                        # Accumulate predictions and targets for this layer
                        pred_tensor = processed_predictions[layer_name]
                        target_tensor = processed_targets[layer_name]
                        
                        if pred_tensor is not None and target_tensor is not None:
                            # Convert to CPU and detach to avoid memory issues
                            if isinstance(pred_tensor, torch.Tensor):
                                pred_tensor = pred_tensor.detach().cpu()
                            if isinstance(target_tensor, torch.Tensor):
                                target_tensor = target_tensor.detach().cpu()
                            
                            self._accumulated_predictions[layer_name].append(pred_tensor)
                            self._accumulated_targets[layer_name].append(target_tensor)
            
            # Handle tensor-based predictions (single layer)
            elif isinstance(processed_predictions, torch.Tensor) and isinstance(processed_targets, torch.Tensor):
                layer_name = 'layer_1'  # Default to layer_1 for single tensor predictions
                
                if layer_name not in self._accumulated_predictions:
                    self._accumulated_predictions[layer_name] = []
                    self._accumulated_targets[layer_name] = []
                
                # Convert to CPU and detach
                pred_tensor = processed_predictions.detach().cpu()
                target_tensor = processed_targets.detach().cpu()
                
                self._accumulated_predictions[layer_name].append(pred_tensor)
                self._accumulated_targets[layer_name].append(target_tensor)
            
            # Log every 10 batches for debugging
            if batch_idx % 10 == 0:
                logger.debug(f"📊 Accumulated metrics from batch {batch_idx}: {self._batch_count} total batches processed")
                
        except Exception as e:
            logger.warning(f"Failed to accumulate batch metrics for batch {batch_idx}: {e}")
    
    def _get_concatenated_metrics(self) -> tuple:
        """
        Concatenate accumulated predictions and targets for metrics calculation.
        
        Returns:
            Tuple of (concatenated_predictions, concatenated_targets)
        """
        if not self._accumulated_predictions or not self._accumulated_targets:
            return {}, {}
        
        concatenated_predictions = {}
        concatenated_targets = {}
        
        for layer_name in self._accumulated_predictions.keys():
            if layer_name in self._accumulated_targets:
                pred_list = self._accumulated_predictions[layer_name]
                target_list = self._accumulated_targets[layer_name]
                
                if pred_list and target_list:
                    try:
                        # Concatenate tensors
                        concatenated_predictions[layer_name] = torch.cat(pred_list, dim=0)
                        concatenated_targets[layer_name] = torch.cat(target_list, dim=0)
                        
                        logger.debug(f"Concatenated {layer_name}: {concatenated_predictions[layer_name].shape} predictions, {concatenated_targets[layer_name].shape} targets")
                    except Exception as e:
                        logger.warning(f"Failed to concatenate {layer_name} metrics: {e}")
        
        return concatenated_predictions, concatenated_targets
    
    @property
    def last_predictions(self):
        """Get accumulated predictions for metrics calculation."""
        concatenated_predictions, _ = self._get_concatenated_metrics()
        return concatenated_predictions
    
    @property
    def last_targets(self):
        """Get accumulated targets for metrics calculation."""
        _, concatenated_targets = self._get_concatenated_metrics()
        return concatenated_targets
    
    @property
    def last_loss_breakdown(self):
        """Get last batch loss breakdown for metrics callback."""
        return getattr(self, '_last_loss_breakdown', {})