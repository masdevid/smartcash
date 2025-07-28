#!/usr/bin/env python3
"""
Refactored training phase management for the unified training pipeline.

This module coordinates training execution using SRP-compliant components
for better maintainability and separation of concerns.
"""

import time
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics

# Import core components
from .core import (
    PhaseOrchestrator,
    TrainingExecutor, 
    ValidationExecutor,
    TrainingCheckpointAdapter,
    ProgressManager
)

logger = get_logger(__name__)


class TrainingPhaseManager:
    """Manages the execution of training phases using SRP-compliant components."""
    
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
        
        # Initialize core components
        self.orchestrator = PhaseOrchestrator(model, model_api, config, progress_tracker)
        self.training_executor = TrainingExecutor(model, config, progress_tracker)
        self.validation_executor = ValidationExecutor(model, config, progress_tracker)
        self.checkpoint_manager = TrainingCheckpointAdapter(model, model_api, config)
        self.progress_manager = ProgressManager(
            progress_tracker, emit_metrics_callback, 
            emit_live_chart_callback, visualization_manager
        )
    
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
            # Set up phase components
            components = self.orchestrator.setup_phase(phase_num, epochs)
            
            # Set single phase mode for proper callbacks and visualization
            is_single_phase = self.config.get('training_mode', 'two_phase') == 'single_phase'
            self.progress_manager.set_single_phase_mode(is_single_phase)
            
            # Initialize variables for tracking best model
            best_metrics = {}
            best_checkpoint_path = None
            
            # Start epoch tracking
            self.progress_manager.start_epoch_tracking(epochs)
            
            # Training loop
            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                
                # Update epoch progress
                self.progress_manager.update_epoch_progress(
                    epoch, epochs, f"Training epoch {epoch + 1}/{epochs}"
                )
                
                # Training
                train_metrics = self.training_executor.train_epoch(
                    components['train_loader'], components['optimizer'], 
                    components['loss_manager'], components['scaler'], 
                    epoch, epochs, phase_num
                )
                
                # Validation
                val_metrics = self.validation_executor.validate_epoch(
                    components['val_loader'], components['loss_manager'], 
                    epoch, epochs, phase_num
                )
                
                # Combine and process metrics
                final_metrics = self._process_epoch_metrics(
                    train_metrics, val_metrics, components['metrics_tracker'], epoch
                )
                
                # Add layer metrics from training executor's last predictions
                layer_metrics = self._compute_layer_metrics()
                final_metrics.update(layer_metrics)
                
                # Ensure all required metrics are present
                self._ensure_required_metrics(final_metrics)
                
                # Emit callbacks and updates
                self._emit_epoch_updates(epoch, phase_num, final_metrics, layer_metrics)
                
                # Check for best model and save checkpoint
                is_best = components['metrics_tracker'].is_best_model()
                if is_best:
                    best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        epoch, final_metrics, phase_num, is_best=True
                    )
                    best_metrics = final_metrics.copy()
                
                # Handle scheduler and early stopping
                self.progress_manager.handle_scheduler_step(components['scheduler'], final_metrics)
                
                should_stop = self.progress_manager.handle_early_stopping(
                    components['early_stopping'], final_metrics, epoch, phase_num
                )
                
                if should_stop:
                    # Ensure we have valid results even if early stopped
                    best_checkpoint_path, best_metrics = self._handle_early_stop_cleanup(
                        epoch, final_metrics, phase_num, best_checkpoint_path, best_metrics
                    )
                    break
                
                # Update epoch progress - normal completion
                self._update_epoch_completion_progress(epoch, epochs, final_metrics, epoch_start_time)
            
            # Final cleanup and result preparation
            return self._prepare_phase_results(
                epoch, best_metrics, best_checkpoint_path, final_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in training phase {phase_num}: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, 
                              metrics_tracker, epoch: int) -> Dict[str, Any]:
        """Process and combine epoch metrics."""
        # Combine metrics efficiently
        final_metrics = {**train_metrics, **val_metrics}
        
        # Add metrics tracker computed metrics
        tracker_metrics = metrics_tracker.compute_epoch_metrics(epoch)
        final_metrics.update(tracker_metrics)
        
        return final_metrics
    
    def _compute_layer_metrics(self) -> Dict[str, float]:
        """Compute layer metrics from training executor's last predictions."""
        layer_metrics = {}
        if (hasattr(self.training_executor, 'last_predictions') and 
            hasattr(self.training_executor, 'last_targets') and
            self.training_executor.last_predictions and 
            self.training_executor.last_targets):
            
            layer_metrics = calculate_multilayer_metrics(
                self.training_executor.last_predictions, 
                self.training_executor.last_targets
            )
        
        return layer_metrics
    
    def _ensure_required_metrics(self, final_metrics: Dict[str, Any]):
        """Ensure all required metrics are present with default values."""
        # Ensure accuracy, precision, recall, f1 are always included (even if zero)
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if f'{layer}_{metric}' not in final_metrics:
                    final_metrics[f'{layer}_{metric}'] = 0.0
                if f'val_{layer}_{metric}' not in final_metrics:
                    final_metrics[f'val_{layer}_{metric}'] = 0.0
    
    def _emit_epoch_updates(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """Emit all epoch updates including callbacks and visualizations."""
        # Emit metrics callback for UI
        self.progress_manager.emit_epoch_metrics(phase_num, epoch + 1, final_metrics)
        
        # Emit live chart data
        self.progress_manager.emit_training_charts(epoch, phase_num, final_metrics, layer_metrics)
        
        # Update visualization manager
        self.progress_manager.update_visualization_manager(epoch, phase_num, final_metrics, layer_metrics)
    
    def _handle_early_stop_cleanup(self, epoch: int, final_metrics: dict, phase_num: int,
                                  best_checkpoint_path: Optional[str], best_metrics: dict) -> tuple:
        """Handle cleanup when early stopping is triggered."""
        # Ensure we have valid results even if early stopped
        if not best_metrics:
            best_metrics = final_metrics.copy()
        if not best_checkpoint_path:
            best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                epoch, final_metrics, phase_num, is_best=True
            )
        
        # Complete epoch tracking due to early stopping
        self.progress_manager.complete_epoch_early_stopping(
            epoch + 1, 
            f"Early stopping triggered - no improvement for {getattr(self.orchestrator, 'early_stopping', {'patience': 'N/A'})}"
        )
        
        return best_checkpoint_path, best_metrics
    
    def _update_epoch_completion_progress(self, epoch: int, epochs: int, 
                                        final_metrics: dict, epoch_start_time: float):
        """Update progress tracking for normal epoch completion."""
        epoch_duration = time.time() - epoch_start_time
        self.progress_manager.update_epoch_progress(
            epoch + 1, epochs,
            f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.1f}s - Loss: {final_metrics.get('train_loss', 0):.4f}"
        )
    
    def _prepare_phase_results(self, epoch: int, best_metrics: dict, 
                              best_checkpoint_path: Optional[str], final_metrics: dict) -> Dict[str, Any]:
        """Prepare final phase results."""
        # Ensure we have valid results
        if not best_metrics:
            best_metrics = final_metrics.copy()
        
        if not best_checkpoint_path:
            best_checkpoint_path = self.checkpoint_manager.ensure_best_checkpoint(
                epoch, final_metrics, 1  # Assume phase 1 if not specified
            )
        
        return {
            'success': True,
            'epochs_completed': epoch + 1,
            'best_metrics': best_metrics,
            'best_checkpoint': best_checkpoint_path,
            'final_metrics': final_metrics,
            'training_completed_successfully': True
        }
    
    def set_single_phase_mode(self, is_single_phase: bool):
        """Set single phase mode flag for proper logging."""
        self.orchestrator.set_single_phase_mode(is_single_phase)
        self.progress_manager.set_single_phase_mode(is_single_phase)