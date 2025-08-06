"""
Training Execution Manager

This module handles the execution of individual training phases,
focusing solely on training and validation loops.
"""

import time
from typing import Dict, Any, Optional

from smartcash.model.training.phases.mixins.callbacks import CallbacksMixin
from smartcash.model.training.phases.mixins.metrics_processing import MetricsProcessingMixin
# Note: Phase setup is now handled by PipelineOrchestrator
from smartcash.model.training.utils.signal_handler import (
    install_training_signal_handlers, register_cleanup_callback, is_shutdown_requested
)
from smartcash.model.training.core import TrainingExecutor, ValidationExecutor, ProgressManager
from smartcash.model.core.checkpoints.checkpoint_manager import create_checkpoint_manager

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class TrainingPhaseExecutor(CallbacksMixin, MetricsProcessingMixin):
    """
    Executes individual training phases with focus on training/validation loops.
    
    Responsibilities:
    - Execute training and validation loops for a single phase
    - Handle epoch-by-epoch training progression
    - Manage metrics processing and callbacks during training
    - Handle early stopping and best model saving
    
    Does NOT handle:
    - High-level pipeline orchestration (delegated to PipelineOrchestrator)
    - Model building/validation (delegated to ModelManager)
    - Phase setup and configuration (delegated to PhaseOrchestrator)
    """
    
    def __init__(self, model, model_api, config, progress_tracker,
                 log_callback=None, metrics_callback=None, live_chart_callback=None,
                 progress_callback=None):
        """
        Initialize training phase executor.
        
        Args:
            model: PyTorch model
            model_api: Model API instance
            config: Training configuration
            progress_tracker: Progress tracking instance
            log_callback: Callback for log messages
            metrics_callback: Callback for metrics updates
            live_chart_callback: Callback for live chart updates
            progress_callback: Callback for progress updates
        """
        super().__init__()
        
        self.model = model
        self.model_api = model_api
        self.config = config
        self.progress_tracker = progress_tracker
        
        # Set callbacks using mixin
        self.set_callbacks(
            log_callback=log_callback,
            metrics_callback=metrics_callback,
            live_chart_callback=live_chart_callback,
            progress_callback=progress_callback
        )
        
        # Note: Phase setup is now handled by PipelineOrchestrator and passed to execute_phase
        
        # Initialize checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(config)
        
        # Install signal handlers for graceful shutdown
        install_training_signal_handlers()
        register_cleanup_callback(self._cleanup_callback)
        
        # Training state
        self.current_phase = None
        
        logger.info("🏃 TrainingPhaseExecutor initialized")
    
    def execute_phase(self, phase_num: int, epochs: int, start_epoch: int = 0, 
                      training_components: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single training phase.
        
        Args:
            phase_num: Training phase number (1 or 2)
            epochs: Total number of epochs for this phase
            start_epoch: Starting epoch (for resume functionality)
            training_components: Pre-configured training components from PipelineOrchestrator
            
        Returns:
            Dictionary containing phase execution results
        """
        self.current_phase = phase_num
        
        try:
            self.emit_log('info', f'🏃 Executing Phase {phase_num}: epochs {start_epoch+1}-{epochs}')
            
            # Use pre-configured training components from PipelineOrchestrator
            if training_components is None:
                raise ValueError(f"training_components must be provided for Phase {phase_num} execution")
            components = training_components
            
            # Execute training loop
            results = self._execute_training_loop(components, phase_num, epochs, start_epoch)
            
            self.emit_log('info', f'✅ Phase {phase_num} execution completed')
            
            return results
            
        except Exception as e:
            self.emit_log('error', f'❌ Phase {phase_num} execution failed: {str(e)}')
            raise
    
    def _execute_training_loop(self, components: Dict[str, Any], phase_num: int, 
                              total_epochs: int, start_epoch: int) -> Dict[str, Any]:
        """Execute the core training loop for a phase."""
        
        # Initialize executors
        training_executor = TrainingExecutor(
            model=self.model,
            config=self.config,
            progress_tracker=self.progress_tracker
        )
        
        validation_executor = ValidationExecutor(
            model=self.model,
            config=self.config,
            progress_tracker=self.progress_tracker,
            phase_num=phase_num
        )
        
        # Initialize progress manager with callbacks
        progress_manager = ProgressManager(
            progress_tracker=self.progress_tracker,
            emit_metrics_callback=self.metrics_callback,
            emit_live_chart_callback=self.live_chart_callback
        )
        
        # Training loop state
        best_metrics = {}
        final_metrics = {}
        
        self.emit_training_start(phase_num, total_epochs, start_epoch)
        
        # Main training loop
        for epoch in range(start_epoch, total_epochs):
            if is_shutdown_requested():
                self.emit_log('warning', '🛑 Shutdown requested, stopping training')
                break
            
            epoch_start_time = time.time()
            
            # Training step
            train_metrics = training_executor.train_epoch(
                train_loader=components['train_loader'],
                optimizer=components['optimizer'],
                loss_manager=components['loss_manager'],
                scaler=components['scaler'],
                epoch=epoch,
                total_epochs=total_epochs,
                phase_num=phase_num
            )
            
            # Validation step
            val_metrics = validation_executor.validate_epoch(
                val_loader=components['val_loader'],
                loss_manager=components.get('loss_manager'),
                epoch=epoch,
                phase_num=phase_num
            )
            
            # Process and combine metrics
            final_metrics = self.process_epoch_metrics(train_metrics, val_metrics, phase_num, epoch)
            
            # Handle scheduler step
            current_lr = progress_manager.handle_scheduler_step(
                components['scheduler'], final_metrics, components['optimizer']
            )
            if current_lr is not None:
                final_metrics['learning_rate'] = current_lr
            
            # Record metrics
            if components.get('metrics_recorder'):
                components['metrics_recorder'].record_epoch(epoch, phase_num, final_metrics)
            
            # Handle early stopping
            should_stop = False
            early_stopping = components.get('early_stopping')
            
            if early_stopping:
                should_stop = progress_manager.handle_early_stopping(
                    early_stopping, final_metrics, epoch, phase_num
                )
            
            # Save checkpoint if this is the best model
            last_checkpoint_path = self._handle_checkpoint_saving(final_metrics, epoch, phase_num, components)
            
            # Update best metrics
            if not best_metrics or final_metrics.get('val_accuracy', 0) > best_metrics.get('val_accuracy', 0):
                best_metrics = final_metrics.copy()
            
            # Emit epoch updates
            self.emit_epoch_updates(epoch, phase_num, final_metrics)
            
            # Update progress
            epoch_time = time.time() - epoch_start_time
            progress_message = (f"Phase {phase_num} - Epoch {epoch+1}/{total_epochs} - "
                              f"Loss: {final_metrics.get('val_loss', 0):.4f} - "
                              f"Acc: {final_metrics.get('val_accuracy', 0):.4f} - "
                              f"LR: {final_metrics.get('learning_rate', 0):.6f} - "
                              f"IoU: {validation_executor.map_calculator.iou_thres:.3f} - "
                              f"Time: {epoch_time:.1f}s")
            
            self.progress_tracker.update_epoch_progress(epoch + 1, total_epochs, progress_message)
            
            # Handle early stopping
            if should_stop:
                self.emit_log('info', f'🛑 Early stopping triggered at epoch {epoch+1}')
                break
        
        # Emit phase completion
        self.emit_phase_completion(phase_num, best_metrics, final_metrics)
        
        # Prepare results
        return {
            'phase': phase_num,
            'final_epoch': epoch,
            'best_metrics': best_metrics,
            'final_metrics': final_metrics,
            'total_epochs': epoch + 1,
            'phase_completed': True,
            'final_checkpoint': last_checkpoint_path # Add this line
        }
    
    def _handle_checkpoint_saving(self, metrics: Dict[str, Any], epoch: int, 
                                 phase_num: int, components: Dict[str, Any]) -> Optional[str]:
        """Handle checkpoint saving logic."""
        try:
            # Save checkpoint with metrics
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                metrics=metrics,
                epoch=epoch,
                phase=phase_num,
                optimizer=components.get('optimizer'),
                scheduler=components.get('scheduler'),
                model_api=self.model_api
            )
            
            # If this was saved as best, emit notification
            if 'best_' in checkpoint_path:
                self.emit_best_model_saved(phase_num, epoch, metrics, checkpoint_path)
            return checkpoint_path
                
        except Exception as e:
            self.emit_log('warning', f'⚠️ Failed to save checkpoint: {str(e)}')
            return None
    
    def handle_phase_transition(self, next_phase_num: int, transition_data: Dict[str, Any] = None):
        """
        Handle phase transitions with state reset.
        
        Args:
            next_phase_num: The phase number we're transitioning to
            transition_data: Optional transition data (unused for clean reset)
        """
        try:
            self.emit_log('info', f'🔄 Phase transition: {self.current_phase} -> {next_phase_num}')
            
            # Phase-specific state management through best metrics manager
            if self.checkpoint_manager:
                best_metrics_manager = self.checkpoint_manager.get_best_metrics_manager()
                
                # Transition to new phase with state reset
                best_metrics_manager.transition_to_phase(next_phase_num, reset_state=True)
                
                self.emit_log('info', f'✅ Phase {next_phase_num} state reset completed')
                self.emit_log('info', '🔄 Scheduler, optimizer, and best checkpoint states are fresh')
            
            self.current_phase = next_phase_num
            
        except Exception as e:
            self.emit_log('error', f'❌ Error during phase transition to {next_phase_num}: {e}')
    
    def _cleanup_callback(self):
        """Cleanup callback for signal handling."""
        self.emit_log('info', '🧹 Cleaning up TrainingPhaseExecutor...')
        
        try:
            # Clean up callback references using mixin
            self.cleanup_callbacks()
            
            # Clear state
            self.current_phase = None
            
        except Exception as e:
            logger.warning(f'⚠️ Error during training phase executor cleanup: {e}')