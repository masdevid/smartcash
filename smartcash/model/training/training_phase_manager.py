#!/usr/bin/env python3
"""
Refactored training phase management for the unified training pipeline.

This module coordinates training execution using SRP-compliant components
for better maintainability and separation of concerns.
"""

import time
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics, filter_phase_relevant_metrics

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
            
            # Training loop - handle resume case where start_epoch might be >= epochs
            if start_epoch >= epochs:
                logger.info(f"🔄 Phase {phase_num} already completed - resume epoch {start_epoch + 1} >= total epochs {epochs}")
                # Return successful completion with empty metrics
                return {
                    'success': True,
                    'epochs_completed': epochs,
                    'best_metrics': {},
                    'best_checkpoint': None,
                    'final_metrics': {},
                    'training_completed_successfully': True,
                    'phase_already_complete': True
                }
            
            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                
                # Calculate display epoch (1-based, adjusted for resume)
                # epoch is the 0-based loop variable, display_epoch is what we show to user
                display_epoch = epoch + 1
                
                # Update epoch progress (show actual epoch numbers, not relative progress)
                current_progress = epoch - start_epoch  # 0-based progress from start (for percentage)
                total_progress = epochs - start_epoch   # total epochs to complete from start
                # Pass actual epoch number for display, but keep relative progress for percentage calculation
                self.progress_manager.update_epoch_progress(
                    display_epoch, epochs, f"Training epoch {display_epoch}/{epochs}",
                    progress_percentage=(current_progress / total_progress * 100) if total_progress > 0 else 0
                )
                
                # Training
                train_metrics = self.training_executor.train_epoch(
                    components['train_loader'], components['optimizer'], 
                    components['loss_manager'], components['scaler'], 
                    epoch, epochs, phase_num, display_epoch=display_epoch
                )
                
                # Optimize train→validation transition
                self._optimize_train_to_validation_transition()
                
                # Validation
                val_metrics = self.validation_executor.validate_epoch(
                    components['val_loader'], components['loss_manager'], 
                    epoch, epochs, phase_num, display_epoch=display_epoch
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
                
                # Filter metrics based on phase before emitting
                filtered_metrics = filter_phase_relevant_metrics(final_metrics, phase_num)
                
                # Emit callbacks and updates with filtered metrics
                self._emit_epoch_updates(epoch, phase_num, filtered_metrics, layer_metrics)
                
                # Save checkpoint every epoch (last.pt for resume)
                # This saves the current training state for proper resume functionality
                # Note: Use complete final_metrics for checkpoint saving, not filtered ones
                self.checkpoint_manager.save_checkpoint(
                    epoch, final_metrics, phase_num, is_best=False
                )
                
                # Check for best model and save best checkpoint
                # Phase 1: Focus on val_loss (minimal) - only layer 1 training
                # Phase 2: Focus on val_map50 (maximal) - full model fine-tuning
                # Note: Use complete final_metrics for best model evaluation
                if phase_num == 1:
                    metric_name = 'val_loss'
                    current_val = final_metrics.get('val_loss', float('inf'))
                    logger.debug(f"🔍 Phase 1 best check: val_loss = {current_val}")
                    is_best = components['metrics_tracker'].is_best_model(
                        metric_name='val_loss', mode='min'
                    )
                else:  # phase_num == 2
                    metric_name = 'val_map50'
                    current_val = final_metrics.get('val_map50', 0.0)
                    logger.debug(f"🔍 Phase 2 best check: val_map50 = {current_val}")
                    is_best = components['metrics_tracker'].is_best_model(
                        metric_name='val_map50', mode='max'
                    )
                
                logger.debug(f"🔍 Best model check result: is_best = {is_best} for {metric_name} = {current_val}")
                
                # FALLBACK: Also check manually for improvement to ensure best checkpoints are saved
                manual_is_best = self._manual_best_check(epoch, phase_num, final_metrics)
                if manual_is_best and not is_best:
                    logger.info(f"🔄 Manual best check overriding tracker decision")
                    is_best = True
                
                if is_best:
                    # Log which metric triggered the best model save
                    if phase_num == 1:
                        metric_value = final_metrics.get('val_loss', 'N/A')
                        logger.info(f"🏆 New best model (Phase {phase_num}): val_loss = {metric_value}")
                    else:
                        metric_value = final_metrics.get('val_map50', 'N/A')
                        logger.info(f"🏆 New best model (Phase {phase_num}): val_map50 = {metric_value}")
                    
                    # Save complete metrics in checkpoint, not filtered ones
                    best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        epoch, final_metrics, phase_num, is_best=True
                    )
                    best_metrics = final_metrics.copy()
                else:
                    # DEBUG: For troubleshooting, log why it wasn't considered best
                    if phase_num == 1:
                        logger.debug(f"🔍 Not best model - Phase 1 val_loss {final_metrics.get('val_loss', 'N/A')} not better than previous")
                    else:
                        logger.debug(f"🔍 Not best model - Phase 2 val_map50 {final_metrics.get('val_map50', 'N/A')} not better than previous")
                
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
                self._update_epoch_completion_progress(epoch, epochs, final_metrics, epoch_start_time, start_epoch)
            
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
        """Emit all epoch updates including callbacks and visualizations.
        
        Note: final_metrics received here should be phase-filtered to show only relevant metrics.
        """
        # Emit metrics callback for UI (using phase-filtered metrics)
        self.progress_manager.emit_epoch_metrics(phase_num, epoch + 1, final_metrics)
        
        # Emit live chart data (using phase-filtered metrics)
        self.progress_manager.emit_training_charts(epoch, phase_num, final_metrics, layer_metrics)
        
        # Update visualization manager (using phase-filtered metrics)
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
                                        final_metrics: dict, epoch_start_time: float, start_epoch: int):
        """Update progress tracking for normal epoch completion."""
        epoch_duration = time.time() - epoch_start_time
        display_epoch = epoch + 1
        current_progress = epoch - start_epoch + 1  # +1 because we just completed this epoch
        total_progress = epochs - start_epoch
        self.progress_manager.update_epoch_progress(
            current_progress, total_progress,
            f"Epoch {display_epoch}/{epochs} completed in {epoch_duration:.1f}s - Loss: {final_metrics.get('train_loss', 0):.4f}"
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
    
    def _optimize_train_to_validation_transition(self):
        """
        Optimize the transition from training to validation to reduce delay.
        
        This method implements several optimizations to minimize the time between
        training and validation phases.
        """
        import torch
        
        try:
            # 1. Force CUDA synchronization now rather than during model.eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 2. Clear GPU cache to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 3. Pre-warm the validation data loader by getting the first batch
            # This avoids the cold-start delay when validation begins
            val_loader = getattr(self, '_val_loader_cache', None)
            if val_loader is not None:
                try:
                    # Pre-fetch first batch (non-blocking)
                    val_iter = iter(val_loader)
                    next(val_iter)  # This warms up the data loader
                except (StopIteration, AttributeError):
                    pass  # Ignore if loader is empty or has issues
            
            logger.debug("🔄 Train→validation transition optimized")
            
        except Exception as e:
            # Don't let optimization errors break training
            logger.debug(f"⚠️ Train→validation optimization failed: {e}")
            pass
    
    def _manual_best_check(self, epoch: int, phase_num: int, final_metrics: Dict[str, float]) -> bool:
        """
        Manual fallback check for best model to ensure best checkpoints are saved.
        
        This provides a backup when the metrics tracker may not detect improvements properly.
        """
        if not hasattr(self, '_manual_best_values'):
            self._manual_best_values = {}
        
        try:
            if phase_num == 1:
                metric_name = 'val_loss'
                current_value = final_metrics.get('val_loss', float('inf'))
                mode = 'min'
            else:
                metric_name = 'val_map50'
                current_value = final_metrics.get('val_map50', 0.0)
                mode = 'max'
            
            # First epoch for this metric
            if metric_name not in self._manual_best_values:
                self._manual_best_values[metric_name] = current_value
                logger.debug(f"🔄 Manual check: First {metric_name} = {current_value}, marking as best")
                return True
            
            # Check for improvement
            previous_best = self._manual_best_values[metric_name]
            
            if mode == 'min':
                is_better = current_value < previous_best
            else:  # mode == 'max'
                is_better = current_value > previous_best
            
            if is_better:
                self._manual_best_values[metric_name] = current_value
                logger.debug(f"🔄 Manual check: {metric_name} improved from {previous_best} to {current_value}")
                return True
            else:
                logger.debug(f"🔄 Manual check: {metric_name} {current_value} not better than {previous_best}")
                return False
                
        except Exception as e:
            logger.debug(f"⚠️ Manual best check failed: {e}")
            return False