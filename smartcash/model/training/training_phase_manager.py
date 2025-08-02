#!/usr/bin/env python3
"""
Refactored training phase management for the unified training pipeline.

This module coordinates training execution using SRP-compliant components
for better maintainability and separation of concerns.
"""

import time
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics, filter_phase_relevant_metrics
from smartcash.model.training.utils.signal_handler import install_training_signal_handlers, register_cleanup_callback, is_shutdown_requested

# Import core components
from .core import (
    PhaseOrchestrator,
    TrainingExecutor, 
    ValidationExecutor,
    ProgressManager
)
from smartcash.model.core.checkpoint_manager import create_checkpoint_manager

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
        self.checkpoint_manager = create_checkpoint_manager(config)
        self.progress_manager = ProgressManager(
            progress_tracker, emit_metrics_callback, 
            emit_live_chart_callback, visualization_manager
        )
        
        # Simple best model tracking
        self._best_metrics = {}
        self._current_phase_num = 1
        
        # Install signal handlers for graceful shutdown
        self.signal_handler = install_training_signal_handlers()
        
        # Register cleanup callbacks
        register_cleanup_callback(self._cleanup_resources)
        register_cleanup_callback(self.progress_manager.cleanup)
        
        # Add DataLoader cleanup
        from smartcash.model.training.data_loader_factory import DataLoaderFactory
        register_cleanup_callback(DataLoaderFactory.cleanup_all)
    
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
            # Set current phase number for tracking
            self._current_phase_num = phase_num
            
            # Set up phase components
            components = self.orchestrator.setup_phase(phase_num, epochs)
            
            # Log comprehensive batch size information
            self._log_training_batch_summary(components, phase_num)
            
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
                # Check for shutdown signal at start of each epoch
                if is_shutdown_requested():
                    logger.info("🛑 Shutdown requested, stopping training gracefully...")
                    return self._prepare_early_shutdown_results(epoch, best_metrics, best_checkpoint_path, final_metrics)
                
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
                    epoch, phase_num, display_epoch
                )
                
                # Combine and process metrics
                final_metrics = self._process_epoch_metrics(
                    train_metrics, val_metrics, components['metrics_recorder'], epoch
                )
                
                # Add layer metrics from training executor's last predictions
                layer_metrics = self._compute_layer_metrics(phase_num)
                
                # Convert training metrics to research-focused format
                final_metrics = self._apply_research_metrics_format(final_metrics, layer_metrics, phase_num)
                
                # Debug: Check metrics consistency
                val_acc = final_metrics.get('val_accuracy', 0.0)
                train_acc = final_metrics.get('train_accuracy', 0.0)
                
                logger.debug(f"🔍 Phase {phase_num} metrics validation:")
                logger.debug(f"    val_accuracy: {val_acc:.6f}")
                logger.debug(f"    train_accuracy: {train_acc:.6f}")
                
                # Check normal train/val differences
                diff = abs(val_acc - train_acc) if train_acc > 0.0 else 0.0
                if diff > 0.1:  # Allow normal train/val differences
                    logger.debug(f"📊 Normal train/validation difference: {diff:.6f}")
                else:
                    logger.debug(f"✅ Metrics consistent: difference = {diff:.6f}")
                
                # Layer metrics merging is now handled in _apply_research_metrics_format method
                # This ensures proper Phase 1 protection against training metrics overwriting validation metrics
                
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
                    model=self.model, metrics=final_metrics, epoch=epoch, phase=phase_num, is_best=False
                )
                
                # Check for best model using research-focused criteria
                from smartcash.model.training.utils.research_metrics import get_research_metrics_manager
                research_metrics_manager = get_research_metrics_manager()
                criteria = research_metrics_manager.get_best_model_criteria(phase_num)
                
                primary_metric = criteria['metric']
                primary_mode = criteria['mode']
                fallback_metric = criteria['fallback_metric']
                fallback_mode = criteria['fallback_mode']
                
                # Try primary metric first
                if primary_metric in final_metrics and final_metrics[primary_metric] > 0.0:
                    is_best = self._is_best_model(
                        metric_name=primary_metric, current_value=final_metrics[primary_metric], mode=primary_mode
                    )
                    metric_name = primary_metric
                    current_val = final_metrics.get(primary_metric, 'N/A')
                    logger.debug(f"🎯 Phase {phase_num} best model selection: {metric_name} = {current_val} ({primary_mode})")
                else:
                    # Use fallback metric
                    is_best = self._is_best_model(
                        metric_name=fallback_metric, current_value=final_metrics.get(fallback_metric, 0), mode=fallback_mode
                    )
                    metric_name = fallback_metric
                    current_val = final_metrics.get(fallback_metric, 'N/A')
                    logger.debug(f"🎯 Phase {phase_num} fallback selection: {metric_name} = {current_val} ({fallback_mode})")
                    logger.warning(f"⚠️ Primary research metric {primary_metric} unavailable, using fallback")
                
                logger.debug(f"🔍 Best model check result: is_best = {is_best} for {metric_name} = {current_val}")
                
                # FALLBACK: Also check manually for improvement to ensure best checkpoints are saved
                manual_is_best = self._manual_best_check(epoch, phase_num, final_metrics)
                if manual_is_best and not is_best:
                    logger.debug(f"🔄 Manual best check overriding tracker decision")
                    is_best = True
                
                if is_best:
                    # Log which metric triggered the best model save
                    if phase_num == 1:
                        metric_value = final_metrics.get('val_accuracy', 'N/A')
                        logger.debug(f"🏆 New best model (Phase {phase_num}): val_accuracy = {metric_value}")
                    else:
                        metric_value = final_metrics.get('val_map50', 'N/A')
                        logger.debug(f"🏆 New best model (Phase {phase_num}): val_map50 = {metric_value}")
                    
                    # Save complete metrics in checkpoint, not filtered ones
                    best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model, metrics=final_metrics, epoch=epoch, phase=phase_num, is_best=True
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
                              metrics_recorder, epoch: int) -> Dict[str, Any]:
        """Process and combine epoch metrics."""
        # Combine metrics efficiently
        final_metrics = {**train_metrics, **val_metrics}
        
        # Add layer-specific metrics if available
        phase_num = getattr(self, '_current_phase_num', 1)
        layer_metrics = self._compute_layer_metrics(phase_num)
        if layer_metrics:
            final_metrics.update(layer_metrics)
            logger.debug(f"Added {len(layer_metrics)} layer metrics to final metrics")
        
        # Add loss breakdown from training executor if available
        if hasattr(self.training_executor, 'last_loss_breakdown'):
            train_loss_breakdown = self.training_executor.last_loss_breakdown
            if train_loss_breakdown:
                # Add individual training loss components directly to final_metrics
                for key, value in train_loss_breakdown.items():
                    final_metrics[f'train_{key}'] = value
                
                # Combine with validation loss breakdown if both exist
                if 'loss_breakdown' in final_metrics:
                    val_loss_breakdown = final_metrics['loss_breakdown']
                    # Add validation loss breakdown components
                    for key, value in val_loss_breakdown.items():
                        final_metrics[f'val_{key}'] = value
                    logger.debug(f"Added validation loss breakdown: {len(val_loss_breakdown)} components")
                
                # Also preserve combined loss_breakdown for compatibility
                combined_loss_breakdown = {}
                for key, value in train_loss_breakdown.items():
                    combined_loss_breakdown[f'train_{key}'] = value
                if 'loss_breakdown' in final_metrics and isinstance(final_metrics['loss_breakdown'], dict):
                    for key, value in final_metrics['loss_breakdown'].items():
                        combined_loss_breakdown[f'val_{key}'] = value
                
                final_metrics['loss_breakdown'] = combined_loss_breakdown
                logger.debug(f"Added training loss breakdown: {len(train_loss_breakdown)} components")
            else:
                logger.debug("Training executor has no loss breakdown available")
        
        # Record metrics with the JSON recorder
        if metrics_recorder and hasattr(metrics_recorder, 'record_epoch'):
            try:
                # Determine current phase number
                phase_num = getattr(self, '_current_phase_num', 1)
                metrics_recorder.record_epoch(epoch + 1, phase_num, final_metrics)
            except Exception as e:
                logger.debug(f"Failed to record metrics: {e}")
        
        # Add learning rate to metrics if available
        if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, '_last_lr'):
            final_metrics['learning_rate'] = self.orchestrator._last_lr
        
        return final_metrics
    
    def _compute_layer_metrics(self, phase_num: int = None) -> Dict[str, float]:
        """Compute phase-aware layer metrics from training executor's last predictions."""
        layer_metrics = {}
        if (hasattr(self.training_executor, 'last_predictions') and 
            hasattr(self.training_executor, 'last_targets') and
            self.training_executor.last_predictions and 
            self.training_executor.last_targets):
            
            # Get all layer metrics first
            all_layer_metrics = calculate_multilayer_metrics(
                self.training_executor.last_predictions, 
                self.training_executor.last_targets
            )
            
            # Filter metrics based on current phase to match validation behavior
            if phase_num == 1:
                # Phase 1: Only include layer_1 metrics
                active_layers = ['layer_1']
            elif phase_num == 2:
                # Phase 2: Include all layer metrics
                active_layers = ['layer_1', 'layer_2', 'layer_3']
            else:
                # Default: Include all layers for unknown phases
                active_layers = ['layer_1', 'layer_2', 'layer_3']
            
            # Filter metrics to only include active layers
            for metric_name, metric_value in all_layer_metrics.items():
                # Check if this metric belongs to an active layer
                is_active_layer = any(layer in metric_name for layer in active_layers)
                if is_active_layer:
                    layer_metrics[metric_name] = metric_value
                else:
                    logger.debug(f"Filtered out training metric {metric_name} for phase {phase_num}")
            
            logger.debug(f"Phase {phase_num} training metrics: {list(layer_metrics.keys())}")
        
        return layer_metrics
    
    def _apply_research_metrics_format(self, final_metrics: Dict[str, float], 
                                     layer_metrics: Dict[str, float], phase_num: int) -> Dict[str, float]:
        """Apply research-focused metrics formatting to training results."""
        from smartcash.model.training.utils.research_metrics import get_research_metrics_manager
        
        # Combine training metrics for processing
        combined_metrics = {**final_metrics}
        
        # Add training layer metrics (remove val_ prefix for training processing)
        # CRITICAL FIX: In Phase 1, don't overwrite validation layer metrics with training metrics
        if phase_num == 1:
            # Only add layer metrics that don't already exist from validation
            for key, value in layer_metrics.items():
                if key not in combined_metrics:
                    combined_metrics[key] = value
                else:
                    from smartcash.common.logger import get_logger
                    logger = get_logger(__name__)
                    logger.debug(f"Phase 1: Keeping validation metric {key}={combined_metrics[key]:.6f} instead of training {value:.6f}")
        else:
            # Phase 2: Use training metrics normally
            for key, value in layer_metrics.items():
                combined_metrics[key] = value
        
        # Create training metrics version (without val_ prefix)
        training_raw_metrics = {}
        for key, value in combined_metrics.items():
            if key.startswith('val_'):
                continue  # Skip validation metrics for training processing
            training_raw_metrics[key] = value
        
        # Get research metrics manager
        research_metrics_manager = get_research_metrics_manager()
        
        # Convert training metrics to research format
        if training_raw_metrics:
            training_research_metrics = research_metrics_manager.standardize_metric_names(
                training_raw_metrics, phase_num, is_validation=False
            )
            # Add training metrics to final metrics
            combined_metrics.update(training_research_metrics)
        
        # Keep validation metrics as they are (already processed by validation executor)
        # Filter out only truly unnecessary metrics, preserve loss_breakdown
        filtered_metrics = self._filter_unnecessary_metrics(combined_metrics, phase_num)
        
        return filtered_metrics
    
    def _filter_unnecessary_metrics(self, metrics: Dict[str, Any], phase_num: int) -> Dict[str, Any]:
        """Filter out only truly unnecessary metrics while preserving loss_breakdown."""
        # Always preserve loss_breakdown regardless of phase
        if 'loss_breakdown' in metrics:
            loss_breakdown = metrics['loss_breakdown']
            logger.debug(f"Preserving loss_breakdown with {len(loss_breakdown)} components")
        
        # Patterns to remove based on phase (be more selective)
        if phase_num == 1:
            # Phase 1: Only remove multi-layer metrics not relevant in Phase 1
            remove_patterns = [
                'layer_2_', 'layer_3_',  # Multi-layer metrics not relevant in Phase 1
                'map50_95', 'map75'  # Only remove detailed mAP metrics, keep map50
            ]
        elif phase_num == 2:
            # Phase 2: Only remove very detailed mAP metrics
            remove_patterns = [
                'map50_95', 'map75'  # Remove detailed mAP, keep map50 for detection info
            ]
        else:
            remove_patterns = ['map50_95', 'map75']
        
        filtered = {}
        for key, value in metrics.items():
            # Always preserve loss_breakdown and its components
            if key == 'loss_breakdown' or any(loss_comp in key for loss_comp in ['box_loss', 'obj_loss', 'cls_loss']):
                filtered[key] = value
                continue
                
            should_remove = any(pattern in key.lower() for pattern in remove_patterns)
            if not should_remove:
                filtered[key] = value
            else:
                logger.debug(f"Filtered unnecessary metric for Phase {phase_num}: {key}")
        
        return filtered
    
    def _emit_phase_completion_metrics_with_status(self, best_metrics: Dict[str, Any], final_metrics: Dict[str, Any], 
                                                 epochs_completed: int, best_checkpoint_path: str, status: str):
        """
        Emit phase completion metrics callback with custom completion status.
        
        Args:
            best_metrics: Best metrics achieved during the phase
            final_metrics: Final metrics from the last epoch
            epochs_completed: Number of epochs completed
            best_checkpoint_path: Path to the best checkpoint saved
            status: Completion status ('success', 'early_stopping', 'early_shutdown')
        """
        if not self.progress_manager.emit_metrics_callback:
            return
        
        try:
            # Determine current phase for callback
            phase_name = 'training_phase_single' if self.progress_manager._is_single_phase else 'training_phase_completion'
            
            # Prepare comprehensive phase completion data
            phase_completion_data = {
                'type': 'phase_completion',
                'phase': phase_name,
                'epochs_completed': epochs_completed,
                'best_metrics': best_metrics.copy(),
                'final_metrics': final_metrics.copy(),
                'best_checkpoint': best_checkpoint_path,
                'completion_status': status
            }
            
            # Add phase-specific summary information
            if 'val_accuracy' in best_metrics:
                phase_completion_data['best_accuracy'] = best_metrics['val_accuracy']
            if 'val_loss' in best_metrics:
                phase_completion_data['best_loss'] = best_metrics['val_loss']
            if 'val_map50' in best_metrics:
                phase_completion_data['best_map50'] = best_metrics['val_map50']
            
            # Log phase completion summary based on status
            status_emoji = {
                'success': '✅',
                'early_stopping': '⏹️',
                'early_shutdown': '🛑'
            }.get(status, '📊')
            
            logger.info(f"📊 Phase completion summary ({status}):")
            logger.info(f"   {status_emoji} Epochs completed: {epochs_completed}")
            if best_checkpoint_path:
                logger.info(f"   🏆 Best checkpoint: {best_checkpoint_path}")
            
            # Log key metrics if available
            if 'val_accuracy' in best_metrics:
                logger.info(f"   🎯 Best accuracy: {best_metrics['val_accuracy']:.4f}")
            if 'val_loss' in best_metrics:
                logger.info(f"   📉 Best loss: {best_metrics['val_loss']:.4f}")
            if 'val_map50' in best_metrics:
                logger.info(f"   🎯 Best mAP@0.5: {best_metrics['val_map50']:.4f}")
            
            # Emit the callback through the progress manager's proper method
            # This ensures consistent callback handling and error checking
            self.progress_manager.emit_epoch_metrics(
                phase_num=0,  # Use 0 for phase completion (not phase-specific)
                epoch=epochs_completed,
                metrics=phase_completion_data,
                loss_breakdown=None
            )
            
            logger.debug(f"✅ Phase completion metrics callback emitted successfully (status: {status})")
            
        except Exception as e:
            logger.warning(f"⚠️ Error emitting phase completion metrics: {str(e)}")
    
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
        # Extract loss breakdown if available from validation metrics
        loss_breakdown = final_metrics.get('loss_breakdown', {})
        self.progress_manager.emit_epoch_metrics(phase_num, epoch + 1, final_metrics, loss_breakdown)
        
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
                model=self.model, metrics=final_metrics, epoch=epoch, phase=phase_num, is_best=True
            )
        
        # Emit phase completion metrics for early stopping
        self._emit_phase_completion_metrics_with_status(
            best_metrics, final_metrics, epoch + 1, 
            best_checkpoint_path, 'early_stopping'
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
            display_epoch, epochs,
            f"Epoch {display_epoch}/{epochs} completed in {epoch_duration:.1f}s - Loss: {final_metrics.get('train_loss', 0):.4f}",
            progress_percentage=(current_progress / total_progress * 100) if total_progress > 0 else 100
        )
    
    def _prepare_phase_results(self, epoch: int, best_metrics: dict, 
                              best_checkpoint_path: Optional[str], final_metrics: dict) -> Dict[str, Any]:
        """Prepare final phase results."""
        # Ensure we have valid results
        if not best_metrics:
            best_metrics = final_metrics.copy()
        
        if not best_checkpoint_path:
            best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model, metrics=final_metrics, epoch=epoch, phase=1, is_best=True
            )
        
        # Emit phase completion metrics callback to show full best model metrics
        self._emit_phase_completion_metrics_with_status(best_metrics, final_metrics, epoch + 1, best_checkpoint_path, 'success')
        
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
    
    def _log_training_batch_summary(self, components: Dict[str, Any], phase_num: int):
        """Log comprehensive batch size summary for training phase."""
        try:
            train_loader = components.get('train_loader')
            val_loader = components.get('val_loader')
            
            if train_loader and val_loader:
                logger.info(f"📊 Phase {phase_num} Batch Configuration Summary:")
                logger.info(f"   • Training Batch Size: {train_loader.batch_size}")
                logger.info(f"   • Validation Batch Size: {val_loader.batch_size}")
                logger.info(f"   • Training Dataset: {len(train_loader.dataset)} samples")
                logger.info(f"   • Validation Dataset: {len(val_loader.dataset)} samples")
                logger.info(f"   • Training Batches per Epoch: {len(train_loader)}")
                logger.info(f"   • Validation Batches per Epoch: {len(val_loader)}")
                
                # Check for memory optimization usage
                config_batch_size = self.config.get('training', {}).get('batch_size', 16)
                actual_batch_size = train_loader.batch_size
                
                if config_batch_size != actual_batch_size:
                    logger.info(f"   • Note: Config batch size ({config_batch_size}) differs from loader batch size ({actual_batch_size})")
                    logger.info(f"   • This may indicate memory optimization or preset override")
                else:
                    logger.info(f"   • Batch size matches configuration: {config_batch_size}")
                    
        except Exception as e:
            logger.debug(f"Could not log batch summary: {e}")
    
    def _cleanup_resources(self):
        """Clean up training resources."""
        try:
            logger.info("🧹 Cleaning up training resources...")
            
            # Clean up model resources
            if hasattr(self.model, 'cleanup'):
                self.model.cleanup()
            
            # PyTorch cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Restore signal handlers
            if hasattr(self, 'signal_handler'):
                self.signal_handler.restore_signal_handlers()
            
            logger.info("✅ Training resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
    def _prepare_early_shutdown_results(self, epoch: int, best_metrics: dict, 
                                      best_checkpoint_path: str, final_metrics: dict) -> Dict[str, Any]:
        """Prepare results when training is interrupted early."""
        # Ensure we have valid results
        if not best_metrics:
            best_metrics = final_metrics.copy() if final_metrics else {}
        
        # Emit phase completion metrics for early shutdown
        if best_metrics:  # Only emit if we have meaningful metrics
            self._emit_phase_completion_metrics_with_status(
                best_metrics, final_metrics or {}, epoch, 
                best_checkpoint_path, 'early_shutdown'
            )
        
        return {
            'success': True,
            'epochs_completed': epoch,
            'best_metrics': best_metrics,
            'best_checkpoint': best_checkpoint_path,
            'final_metrics': final_metrics if final_metrics else {},
            'training_completed_successfully': False,
            'early_shutdown': True,
            'reason': 'User interruption (Ctrl+C)'
        }
    
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
    
    def _manual_best_check(self, _epoch: int, _phase_num: int, final_metrics: Dict[str, float]) -> bool:
        """
        Manual fallback check for best model to ensure best checkpoints are saved.
        
        This provides a backup when the metrics tracker may not detect improvements properly.
        """
        if not hasattr(self, '_manual_best_values'):
            self._manual_best_values = {}
        
        try:
            metric_name = 'val_accuracy'
            current_value = final_metrics.get('val_accuracy', 0.0)
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
    
    def _is_best_model(self, metric_name: str, current_value: float, mode: str = 'max') -> bool:
        """
        Simple best model tracking.
        
        Args:
            metric_name: Name of the metric to track
            current_value: Current value of the metric
            mode: 'max' for higher is better, 'min' for lower is better
            
        Returns:
            True if this is the best value seen so far
        """
        if metric_name not in self._best_metrics:
            self._best_metrics[metric_name] = current_value
            return True
        
        best_value = self._best_metrics[metric_name]
        
        if mode == 'max':
            is_better = current_value > best_value
        else:  # mode == 'min'
            is_better = current_value < best_value
        
        if is_better:
            self._best_metrics[metric_name] = current_value
            return True
        
        return False
    
    def handle_phase_transition(self, new_phase_num: int, components: Dict[str, Any]):
        """Handle transition to a new training phase and update metrics recorder."""
        logger.info(f"Handling phase transition to phase {new_phase_num}")
        
        # Update current phase tracking
        old_phase = getattr(self, '_current_phase_num', 1)
        self._current_phase_num = new_phase_num
        
        # Update metrics recorder if available
        metrics_recorder = components.get('metrics_recorder')
        if metrics_recorder and hasattr(metrics_recorder, 'update_phase'):
            try:
                metrics_recorder.update_phase(new_phase_num)
                logger.info(f"Updated metrics recorder from phase {old_phase} to phase {new_phase_num}")
            except Exception as e:
                logger.warning(f"Failed to update metrics recorder phase: {e}")
        
        # Log phase transition for debugging
        logger.info(f"Phase transition completed: {old_phase} → {new_phase_num}")