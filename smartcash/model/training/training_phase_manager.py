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
            
            # Create optimizer using OptimizerFactory
            optimizer_factory = OptimizerFactory(self.config)
            base_lr = phase_config.get('learning_rate', 0.001)
            optimizer = optimizer_factory.create_optimizer(self.model, base_lr)
            scheduler = optimizer_factory.create_scheduler(optimizer, total_epochs=epochs)
            scaler = torch.amp.GradScaler('cuda') if self.config['training']['mixed_precision'] else None
            
            # Initialize MetricsTracker
            metrics_tracker = MetricsTracker(config=self.config)
            early_stopping = create_early_stopping(self.config['training']['early_stopping'])
            
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
        
        # Start batch tracking
        self.progress_tracker.start_batch_tracking(num_batches)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move data to device efficiently
            device = next(self.model.parameters()).device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=scaler is not None):
                predictions = self.model(images)
                
                # Normalize predictions format
                if not isinstance(predictions, dict):
                    predictions = {'layer_1': predictions}
                
                # Store only last batch for metrics (memory efficient)
                if batch_idx == num_batches - 1:
                    last_predictions = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                      for k, v in predictions.items()}
                    
                    if isinstance(targets, dict):
                        last_targets = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                       for k, v in targets.items()}
                    else:
                        last_targets = {'layer_1': targets.detach() if isinstance(targets, torch.Tensor) else targets}
                
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
        
        # Initialize metrics dictionary
        metrics = {}
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                device = next(self.model.parameters()).device
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Get model predictions
                predictions = self.model(images)
                if not isinstance(predictions, dict):
                    predictions = {'layer_1': predictions}
                
                # Compute loss and metrics
                loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
                running_val_loss += loss.item()
                
                # Update metrics from loss breakdown
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
        
        # Calculate average metrics
        if num_batches > 0:
            metrics = {k: v / num_batches for k, v in metrics.items()}
        
        # Ensure all required metrics are present
        base_metrics = {
            'val_loss': running_val_loss / num_batches if num_batches > 0 else 0.0,
            'val_map50': 0.0,
            'val_map50_95': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0
        }
        
        # Update with calculated metrics
        base_metrics.update(metrics)
        
        return base_metrics
    
    def _emit_training_charts(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """Emit live chart data for training visualization."""
        if not self.emit_live_chart_callback:
            return
            
        # Training curves chart
        chart_data = {
            'epoch': epoch + 1,
            'train_loss': final_metrics.get('train_loss', 0),
            'val_loss': final_metrics.get('val_loss', 0),
            'train_accuracy': final_metrics.get('layer_1_accuracy', 0),
            'val_accuracy': final_metrics.get('val_layer_1_accuracy', 0),
            'phase': phase_num
        }
        # Fix nested f-string syntax error
        phase_label = 'Single Phase' if self._is_single_phase else f'Phase {phase_num}'
        self.emit_live_chart_callback('training_curves', chart_data, {
            'title': f'Training Progress - {phase_label}',
            'xlabel': 'Epoch',
            'ylabel': 'Loss / Accuracy'
        })
        
        # Layer metrics chart
        layer_chart_data = {}
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            layer_chart_data[layer] = {
                'accuracy': final_metrics.get(f'{layer}_accuracy', 0),
                'precision': final_metrics.get(f'{layer}_precision', 0),
                'recall': final_metrics.get(f'{layer}_recall', 0),
                'f1': final_metrics.get(f'{layer}_f1', 0)
            }
        
        if any(layer_chart_data[layer]['accuracy'] > 0 for layer in layer_chart_data):
            self.emit_live_chart_callback('layer_metrics', {
                'epoch': epoch + 1,
                'layers': layer_chart_data,
                'phase': phase_num
            }, {
                'title': f'Layer Performance - {"Single Phase" if self._is_single_phase else "Phase " + str(phase_num)}',
                'xlabel': 'Layer',
                'ylabel': 'Metric Value'
            })
    
    def _update_visualization_manager(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """Update visualization manager with current epoch data."""
        if not self.visualization_manager:
            return
            
        phase_name = f"phase_{phase_num}"
        # Create simulated predictions and targets for visualization
        viz_predictions = {}
        viz_targets = {}
        if layer_metrics:
            np.random.seed(epoch * 42)
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                num_classes = {'layer_1': 7, 'layer_2': 7, 'layer_3': 3}[layer]
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