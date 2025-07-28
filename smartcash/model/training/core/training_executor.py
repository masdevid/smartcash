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
        self.prediction_processor = PredictionProcessor(config)
        
        # State for metrics calculation
        self._last_predictions = None
        self._last_targets = None
    
    def train_epoch(self, train_loader, optimizer, loss_manager, scaler, 
                   epoch: int, total_epochs: int, phase_num: int) -> Dict[str, float]:
        """
        Train for one epoch with optimized progress tracking.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
            loss_manager: Loss manager instance
            scaler: Mixed precision scaler
            epoch: Current epoch number
            total_epochs: Total number of epochs
            phase_num: Current phase number
            
        Returns:
            Dictionary containing training metrics
        """
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
            
            # Process batch
            loss, predictions, processed_predictions, processed_targets = self._process_training_batch(
                images, targets, loss_manager, batch_idx, num_batches, phase_num
            )
            
            # Store only last batch for metrics (memory efficient)
            if batch_idx == num_batches - 1:
                last_predictions = processed_predictions
                last_targets = processed_targets
            
            # Backward pass
            self._backward_pass(loss, optimizer, scaler)
            
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
    
    def _process_training_batch(self, images, targets, loss_manager, batch_idx, num_batches, phase_num):
        """Process a single training batch."""
        # Move data to device efficiently
        device = next(self.model.parameters()).device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=hasattr(self, 'scaler') and self.scaler is not None):
            predictions = self.model(images)
            
            # Normalize predictions format
            predictions = self.prediction_processor.normalize_training_predictions(
                predictions, phase_num, batch_idx
            )
            
            # Process predictions for metrics (only last batch)
            processed_predictions = None
            processed_targets = None
            if batch_idx == num_batches - 1:
                processed_predictions, processed_targets = self.prediction_processor.process_for_metrics(
                    predictions, targets, images, device
                )
            
            # Calculate loss
            loss, _ = loss_manager.compute_loss(predictions, targets, images.shape[-1])
        
        return loss, predictions, processed_predictions, processed_targets
    
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
    
    @property
    def last_predictions(self):
        """Get last batch predictions for metrics calculation."""
        return self._last_predictions
    
    @property
    def last_targets(self):
        """Get last batch targets for metrics calculation."""
        return self._last_targets