"""
Training phase manager for executing training phases.

Manages the execution of training phases using SRP-compliant components.
"""

import time
from typing import Dict, Any, Optional

from .base import BasePhaseManager
from .mixins.metrics_processing import MetricsProcessingMixin
from .mixins.progress_tracking import ProgressTrackingMixin
from .orchestrator import PhaseOrchestrator

from smartcash.model.training.utils.signal_handler import (
    install_training_signal_handlers, register_cleanup_callback, is_shutdown_requested
)

# Import core components
from smartcash.model.training.core import (
    TrainingExecutor, 
    ValidationExecutor,
    ProgressManager
)
from smartcash.model.core.checkpoints.checkpoint_manager import create_checkpoint_manager


class TrainingPhaseManager(BasePhaseManager, MetricsProcessingMixin, ProgressTrackingMixin):
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
        super().__init__(model, model_api, config, progress_tracker)
        
        # Callback functions
        self.emit_metrics_callback = emit_metrics_callback
        self.emit_live_chart_callback = emit_live_chart_callback
        self.visualization_manager = visualization_manager
        
        # Initialize orchestrator for setup operations
        self.orchestrator = PhaseOrchestrator(model, model_api, config, progress_tracker)
        
        # Initialize core components
        self.progress_manager = ProgressManager(
            progress_tracker=progress_tracker,
            visualization_manager=visualization_manager,
            emit_metrics_callback=emit_metrics_callback,
            emit_live_chart_callback=emit_live_chart_callback
        )
        
        # Initialize checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(config)
        
        # Install signal handlers for graceful shutdown
        install_training_signal_handlers()
        register_cleanup_callback(self._cleanup_callback)
        
        self.logger.info("🎬 TrainingPhaseManager initialized")
    
    def setup_phase(self, phase_num: int, epochs: int, save_best_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Set up training phase (delegates to orchestrator).
        
        Args:
            phase_num: Training phase number
            epochs: Total number of epochs
            save_best_path: Path to save best model
            
        Returns:
            Dictionary containing setup components
        """
        return self.orchestrator.setup_phase(phase_num, epochs, save_best_path)
    
    def execute_phase(self, phase_num: int, epochs: int, start_epoch: int = 0, 
                     save_best_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a training phase.
        
        Args:
            phase_num: Training phase number
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resume)
            save_best_path: Path to save best model
            
        Returns:
            Dictionary containing training results
        """
        self.logger.info(f"🎬 Executing Phase {phase_num}: epochs {start_epoch+1}-{epochs}")
        
        # Set up phase components
        components = self.setup_phase(phase_num, epochs, save_best_path)
        
        # Set up progress tracking
        self.progress_manager.start_epoch_tracking(epochs)
        self.progress_manager.set_single_phase_mode(self.is_single_phase())
        
        # Initialize training and validation executors
        training_executor = TrainingExecutor(
            model=self.model,
            config=self.config,
            progress_tracker=self.progress_tracker
        )
        
        validation_executor = ValidationExecutor(
            model=self.model,
            config=self.config,
            progress_tracker=self.progress_tracker
        )
        
        # Log training batch summary
        self._log_training_batch_summary(components, phase_num)
        
        # Training loop
        best_metrics = {}
        
        for epoch in range(start_epoch, epochs):
            if is_shutdown_requested():
                self.logger.info("🛑 Shutdown requested, stopping training")
                break
            
            epoch_start_time = time.time()
            
            # Training step
            train_metrics = training_executor.train_epoch(
                train_loader=components['train_loader'],
                optimizer=components['optimizer'],
                loss_manager=components['loss_manager'],
                scaler=components['scaler'],
                epoch=epoch,
                total_epochs=epochs,
                phase_num=phase_num
            )
            
            # Validation step  
            val_metrics = validation_executor.validate_epoch(
                val_loader=components['val_loader'],
                loss_manager=components.get('loss_manager'),
                epoch=epoch,
                phase_num=phase_num
            )
            
            # Process metrics
            final_metrics = self.process_epoch_metrics(
                train_metrics, val_metrics, phase_num, epoch
            )
            
            # Handle scheduler step
            current_lr = self.progress_manager.handle_scheduler_step(
                components['scheduler'], final_metrics, components['optimizer']
            )
            if current_lr is not None:
                final_metrics['learning_rate'] = current_lr
            
            # Record metrics
            if components['metrics_recorder']:
                components['metrics_recorder'].record_epoch(epoch, phase_num, final_metrics)
            
            # Check early stopping
            early_stopping = components['early_stopping']
            should_stop = False
            
            if early_stopping:
                should_stop = self.progress_manager.handle_early_stopping(
                    early_stopping, final_metrics, epoch, phase_num
                )
            
            # Update visualization and progress
            self.progress_manager.update_visualization_manager(
                epoch=epoch,
                phase_num=phase_num,
                final_metrics=final_metrics,
                layer_metrics={}
            )
            
            # Emit updates with all required parameters
            self.emit_epoch_updates(epoch, phase_num, final_metrics)
            
            # Handle early stopping
            if should_stop:
                self.handle_early_stop_cleanup(epoch, final_metrics, phase_num,
                                              early_stopping, self.progress_manager)
                break
            
            # Update best metrics
            if not best_metrics or final_metrics.get('val_accuracy', 0) > best_metrics.get('val_accuracy', 0):
                best_metrics = final_metrics.copy()
            
            # Update progress
            epoch_time = time.time() - epoch_start_time
            progress_message = f"Phase {phase_num} - Epoch {epoch+1}/{epochs} - " \
                             f"Loss: {final_metrics.get('val_loss', 0):.4f} - " \
                             f"Acc: {final_metrics.get('val_accuracy', 0):.4f} - " \
                             f"Time: {epoch_time:.1f}s"
            
            self.update_epoch_completion_progress(
                epoch, epochs, progress_message, self.progress_manager
            )
        
        # Emit final metrics
        self._emit_phase_completion_metrics_with_status(best_metrics, final_metrics, phase_num)
        
        # Prepare results
        results = self._prepare_phase_results(epochs-1, best_metrics, final_metrics, phase_num)
        
        self.logger.info(f"✅ Phase {phase_num} execution completed")
        return results
    
    def run_training_phase(self, phase_num: int, epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Legacy method name for backward compatibility.
        
        Args:
            phase_num: Training phase number
            epochs: Number of epochs to train
            start_epoch: Starting epoch
            
        Returns:
            Dictionary containing training results
        """
        return self.execute_phase(phase_num, epochs, start_epoch)
    
    def emit_epoch_updates(self, epoch: int, phase_num: int, metrics: Dict[str, Any]):
        """Emit epoch updates to all registered callbacks."""
        phase_str = f'phase_{phase_num}'
        
        # Metrics callback
        if self.emit_metrics_callback:
            try:
                self.emit_metrics_callback(phase_str, epoch, metrics)
            except Exception as e:
                self.logger.warning(f"Error in metrics callback: {e}")
                # Fallback for older callback signatures
                try:
                    self.emit_metrics_callback(phase_str, metrics)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback metrics callback failed: {fallback_e}")

        # Live chart callback
        if self.emit_live_chart_callback:
            try:
                # Ensure the message is a string
                self.emit_live_chart_callback('info', f"Epoch {epoch} metrics updated", metrics)
            except Exception as e:
                self.logger.warning(f"Error in live chart callback: {e}")
                # Fallback for older callback signatures
                try:
                    self.emit_live_chart_callback(phase_str, metrics)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback live chart callback failed: {fallback_e}")
    def _emit_phase_completion_metrics_with_status(self, best_metrics: Dict[str, Any], final_metrics: Dict[str, Any], phase_num: int):
        """Emit phase completion metrics with status information."""
        # Add phase completion status
        completion_metrics = {
            **final_metrics,
            'phase_completed': phase_num,
            'phase_status': 'completed',
            'best_val_accuracy': best_metrics.get('val_accuracy', 0),
            'best_val_loss': best_metrics.get('val_loss', 0)
        }
        
        # Get current epoch and format phase string
        current_epoch = final_metrics.get('epoch', 0)
        phase_str = f'phase_{phase_num}'
        
        # Call metrics callback if available
        if self.emit_metrics_callback:
            self.emit_metrics_callback(phase_str, current_epoch, completion_metrics)
        
        # Call live chart callback if available
        if hasattr(self, 'emit_live_chart_callback') and self.emit_live_chart_callback:
            self.emit_live_chart_callback(phase_str, completion_metrics)
            
    
    def _prepare_phase_results(self, final_epoch: int, best_metrics: dict, final_metrics: dict, phase_num: int) -> Dict[str, Any]:
        """Prepare phase results dictionary."""
        return {
            'phase': phase_num,
            'final_epoch': final_epoch,
            'best_metrics': best_metrics,
            'final_metrics': final_metrics,
            'total_epochs': final_epoch + 1,
            'phase_completed': True
        }
    
    def handle_phase_transition(self, next_phase_num: int, transition_data: Dict[str, Any] = None):
        """
        Handle phase transitions with state reset (scheduler, optimizer, best checkpoint).
        
        Args:
            next_phase_num: The phase number we're transitioning to
            transition_data: Optional transition data (unused for clean reset)
        """
        try:
            self.logger.info(f"🔄 Phase transition: transitioning to Phase {next_phase_num}")
            
            # Phase-specific state management through best metrics manager
            if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager:
                best_metrics_manager = self.checkpoint_manager.get_best_metrics_manager()
                
                # Transition to new phase with state reset
                # This resets scheduler, optimizer, and best checkpoint state for the new phase
                best_metrics_manager.transition_to_phase(next_phase_num, reset_state=True)
                
                self.logger.info(f"✅ Phase {next_phase_num} state reset completed")
                self.logger.info("🔄 Scheduler, optimizer, and best checkpoint states will be fresh for this phase")
            else:
                self.logger.warning("⚠️ No checkpoint manager available for phase transition")
                
        except Exception as e:
            self.logger.error(f"❌ Error during phase transition to {next_phase_num}: {e}")
    
    def _log_training_batch_summary(self, components: Dict[str, Any], phase_num: int):
        """Log training batch summary for debugging."""
        train_loader = components.get('train_loader')
        val_loader = components.get('val_loader')
        
        if train_loader and val_loader:
            self.logger.info(f"📊 Phase {phase_num} Dataset Summary:")
            self.logger.info(f"   Training batches: {len(train_loader)}")
            self.logger.info(f"   Validation batches: {len(val_loader)}")
            
            # Get batch size from first batch if available
            try:
                first_batch = next(iter(train_loader))
                if isinstance(first_batch, (list, tuple)) and len(first_batch) > 0:
                    batch_size = len(first_batch[0]) if hasattr(first_batch[0], '__len__') else 'unknown'
                    self.logger.info(f"   Batch size: {batch_size}")
            except:
                pass
    
    def _cleanup_callback(self):
        """Cleanup callback for signal handling."""
        self.logger.info("🧹 Cleaning up TrainingPhaseManager...")
        
        if hasattr(self, 'progress_manager') and self.progress_manager:
            self.progress_manager.cleanup()
        
        # Clear callback references
        self.emit_metrics_callback = None
        self.emit_live_chart_callback = None