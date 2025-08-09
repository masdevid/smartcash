#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/progress_tracker.py

Progress tracking utilities for unified training pipeline.
"""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressData:
    """Data structure for progress information."""
    overall_progress: float = 0.0
    epoch_progress: float = 0.0
    batch_progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    phase: str = "training"
    message: str = ""
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class TrainingProgressTracker:
    """
    Progress tracker for the unified training pipeline with 3-level progress:
    1. Overall: Preparation -> Build Model -> Validate Model -> Start Train Phase 1 -> [Start Train Phase 2] -> Finalize
    2. Epoch: Current epoch progress (with early stopping support)
    3. Batch: Current batch progress within epoch
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None, verbose: bool = True, training_mode: str = 'two_phase', ui_components: Optional[Dict[str, Any]] = None, metrics_callback: Optional[Callable] = None):
        """Initialize the progress tracker.
        
        Args:
            progress_callback: Optional callback function for progress updates
            verbose: Whether to enable verbose logging
            training_mode: 'single_phase' or 'two_phase' to determine total phases
            ui_components: Dictionary of UI components (for UI integration)
            metrics_callback: Optional callback function for metrics updates
        """
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.training_mode = training_mode
        self.ui_components = ui_components or {}
        self.metrics_callback = metrics_callback
        
        # Progress data structure
        self.current_progress = ProgressData()
        
        # Overall phases - 5 phases for single, 6 phases for two_phase
        if training_mode == 'single_phase':
            self.phases = [
                'preparation',
                'build_model', 
                'validate_model',
                'training_phase_1',
                'finalize'
            ]
        else:  # two_phase
            self.phases = [
                'preparation',
                'build_model', 
                'validate_model',
                'training_phase_1',
                'training_phase_2',
                'finalize'
            ]
        
        # Overall progress tracking
        self.current_phase = None
        self.current_phase_index = -1
        self.phase_start_time = None
        self.phase_results = {}
        self.pipeline_start_time = time.time()
        
        # Epoch progress tracking
        self.current_epoch = 0
        self.total_epochs = 0
        self.epoch_completed = False
        self.early_stopping_triggered = False
        
        # Batch progress tracking
        self.current_batch = 0
        self.total_batches = 0
        self.batch_progress_active = False
        
        # Timing statistics
        self.timing_stats = {
            'total_time': 0.0,
            'epoch_times': [],
            'batch_times': [],
            'avg_epoch_time': 0.0,
            'avg_batch_time': 0.0
        }
        
        # Operation state for generic progress tracking
        self.current_operation = ""
        self.operation_steps = 0
        self.current_step = 0
        self.substep_current = 0
        self.substep_total = 0
    
    def start_phase(self, phase_name: str, total_steps: int, description: str = ""):
        """Start a new phase."""
        self.current_phase = phase_name
        self.current_phase_index = self.phases.index(phase_name) if phase_name in self.phases else -1
        self.phase_start_time = time.time()
        
        logger.info(f"🚀 Starting {phase_name.replace('_', ' ').title()}")
        logger.info(f"   {description}")
        
        if self.progress_callback:
            self.progress_callback(phase_name, 0, total_steps, f"Starting {phase_name.replace('_', ' ').title()}")

    def update_overall_progress(self, current_phase_name: str, current_step: int, total_steps: int):
        """Update the overall progress bar with named steps."""
        if self.progress_callback:
            # The overall progress bar is managed by the executor, not the orchestrator
            # This method is called by the executor to update its overall progress
            # The orchestrator's progress_tracker is a sub-tracker
            self.progress_callback('overall', current_step, total_steps, current_phase_name)
    
    def start_epoch_tracking(self, total_epochs: int):
        """Start epoch progress tracking."""
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.epoch_completed = False
        self.early_stopping_triggered = False
        
        if self.progress_callback:
            self.progress_callback('epoch', 0, total_epochs, f"Starting epoch training ({total_epochs} epochs)")
    
    def update_epoch_progress(self, current_epoch: int, total_epochs: int = None, message: str = "", relative_current: int = None):
        """Update epoch progress with option for custom relative position."""
        self.current_epoch = current_epoch
        if total_epochs:
            self.total_epochs = total_epochs
            
        if self.progress_callback:
            # Use relative_current for progress bar position if provided, otherwise use current_epoch
            progress_position = relative_current if relative_current is not None else current_epoch
            self.progress_callback('epoch', progress_position, self.total_epochs, message, epoch=current_epoch)
    
    def complete_epoch_early_stopping(self, final_epoch: int, message: str = "Early stopping triggered"):
        """Complete epoch tracking due to early stopping."""
        self.early_stopping_triggered = True
        self.current_epoch = final_epoch
        
        # Set progress to 100% when early stopping
        if self.progress_callback:
            self.progress_callback('epoch', 100, 100, message)
    
    def start_batch_tracking(self, total_batches: int):
        """Start batch progress tracking."""
        self.total_batches = total_batches
        self.current_batch = 0
        self.batch_progress_active = True
        
        if self.progress_callback:
            self.progress_callback('batch', 0, total_batches, f"Starting batch processing ({total_batches} batches)")
    
    def update_batch_progress(self, current_batch: int, total_batches: int = None, message: str = "", loss: float = None, **kwargs):
        """Update batch progress."""
        if not self.batch_progress_active:
            return
            
        self.current_batch = current_batch
        if total_batches:
            self.total_batches = total_batches
        
        # Add loss to message if provided
        if loss is not None:
            message = f"{message} (Loss: {loss:.4f})" if message else f"Loss: {loss:.4f}"
            
        if self.progress_callback:
            # Pass through additional kwargs (like epoch) to the callback
            self.progress_callback('batch', current_batch, self.total_batches, message, loss=loss, **kwargs)
    
    def complete_batch_tracking(self):
        """Complete batch progress tracking."""
        self.batch_progress_active = False
        
        if self.progress_callback:
            self.progress_callback('batch', 100, 100, "Batch processing completed")
    
    def complete_phase(self, result: Dict[str, Any]):
        """Complete the current phase."""
        if not self.current_phase:
            return
            
        duration = time.time() - self.phase_start_time if self.phase_start_time else 0
        result['duration'] = duration
        self.phase_results[self.current_phase] = result
        
        success = result.get('success', False)
        status = "✅" if success else "❌"
        phase_display = self.current_phase.replace('_', ' ').title()
        
        logger.info(f"{status} {phase_display} completed in {duration:.1f}s")
        
        if self.progress_callback:
            if success:
                # Successful completion - report 100%
                message = f"✅ {phase_display} completed"
                self.progress_callback(self.current_phase, 100, 100, message)
            else:
                # Failed completion - report partial progress to avoid confusion
                error_msg = result.get('error', 'Unknown error')
                message = f"❌ {phase_display} failed: {error_msg}"
                # Report 99% instead of 100% to indicate incomplete/failed state
                self.progress_callback(self.current_phase, 99, 100, message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete pipeline summary."""
        total_duration = time.time() - self.pipeline_start_time
        phases_completed = len([r for r in self.phase_results.values() if r.get('success', False)])
        
        return {
            'total_duration': total_duration,
            'phases_completed': phases_completed,
            'total_phases': len(self.phases),
            'success': phases_completed == len(self.phases),
            'phase_results': self.phase_results,
            'phases': self.phases
        }
    
    def get_phase_result(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific phase."""
        return self.phase_results.get(phase_name)
    
    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase was completed successfully."""
        result = self.phase_results.get(phase_name)
        return result is not None and result.get('success', False)
    
    def get_current_phase_info(self) -> Dict[str, Any]:
        """Get information about current phase."""
        if not self.current_phase:
            return {}
            
        return {
            'phase': self.current_phase,
            'phase_index': self.current_phase_index,
            'phase_display': self.current_phase.replace('_', ' ').title(),
            'duration': time.time() - self.phase_start_time if self.phase_start_time else 0
        }
    
    # Training-specific methods (from TrainingProgressBridge)
    def start_training(self, total_epochs: int, total_batches_per_epoch: int = None,
                      operation_name: str = "Model Training") -> None:
        """Initialize training progress tracking.
        
        Args:
            total_epochs: Total number of epochs
            total_batches_per_epoch: Total batches per epoch (optional)
            operation_name: Display name for operation
        """
        self.current_progress.total_epochs = total_epochs
        self.current_progress.total_batches = total_batches_per_epoch or 0
        self.current_progress.phase = "training"
        self.current_progress.message = f"🚀 Starting {operation_name}..."
        
        # Initialize UI progress tracker
        self._update_ui_progress_tracker("show", operation_name, ["Training", "Validation", "Completion"])
        
        # Send initial progress update
        self._send_progress_update()
    
    def start_epoch(self, epoch: int, phase: str = "training") -> None:
        """Mark start of new epoch.
        
        Args:
            epoch: Current epoch number (0-based)
            phase: Current phase (training/validation)
        """
        self.current_progress.current_epoch = epoch + 1  # 1-based for display
        self.current_progress.phase = phase
        self.current_progress.batch_progress = 0.0
        self.current_progress.current_batch = 0
        
        # Calculate overall progress
        self.current_progress.overall_progress = (epoch / self.current_progress.total_epochs) * 100
        
        phase_emoji = "🔥" if phase == "training" else "📊"
        self.current_progress.message = f"{phase_emoji} Epoch {self.current_progress.current_epoch}/{self.current_progress.total_epochs} - {phase.capitalize()}"
        
        self._send_progress_update()
    
    def update_batch(self, batch_idx: int, total_batches: int, loss: float = None,
                    metrics: Dict[str, float] = None) -> None:
        """Update batch progress.
        
        Args:
            batch_idx: Current batch index (0-based)
            total_batches: Total batches in epoch
            loss: Current batch loss
            metrics: Additional metrics
        """
        self.current_progress.current_batch = batch_idx + 1  # 1-based display
        self.current_progress.total_batches = total_batches
        
        # Calculate batch progress within epoch
        self.current_progress.batch_progress = ((batch_idx + 1) / total_batches) * 100
        
        # Calculate epoch progress (training + validation combined)
        epoch_base_progress = (self.current_progress.current_epoch - 1) / self.current_progress.total_epochs
        epoch_increment = (1 / self.current_progress.total_epochs) * (self.current_progress.batch_progress / 100)
        
        if self.current_progress.phase == "validation":
            # Validation is second half of epoch progress
            epoch_increment = (1 / self.current_progress.total_epochs) * (0.5 + (self.current_progress.batch_progress / 100) * 0.5)
        elif self.current_progress.phase == "training":
            # Training is first half of epoch progress
            epoch_increment = (1 / self.current_progress.total_epochs) * ((self.current_progress.batch_progress / 100) * 0.5)
        
        self.current_progress.overall_progress = (epoch_base_progress + epoch_increment) * 100
        
        # Update message with batch info
        phase_emoji = "🔥" if self.current_progress.phase == "training" else "📊"
        loss_str = f", Loss: {loss:.4f}" if loss is not None else ""
        self.current_progress.message = f"{phase_emoji} Epoch {self.current_progress.current_epoch}/{self.current_progress.total_epochs} - Batch {self.current_progress.current_batch}/{total_batches}{loss_str}"
        
        # Update metrics
        if metrics:
            self.current_progress.metrics.update(metrics)
        if loss is not None:
            self.current_progress.metrics['current_loss'] = loss
        
        # Send updates
        self._send_progress_update()
        if self.current_progress.metrics:
            self._send_metrics_update()
    
    def complete_epoch(self, epoch_metrics: Dict[str, float] = None) -> None:
        """Mark completion of epoch.
        
        Args:
            epoch_metrics: Metrics from completed epoch
        """
        # Update timing stats
        if hasattr(self, 'epoch_start_time') and self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.timing_stats['epoch_times'].append(epoch_time)
            self.timing_stats['avg_epoch_time'] = sum(self.timing_stats['epoch_times']) / len(self.timing_stats['epoch_times'])
        
        # Update progress
        self.current_progress.epoch_progress = 100.0
        self.current_progress.batch_progress = 100.0
        
        # Update metrics with epoch results
        if epoch_metrics:
            self.current_progress.metrics.update(epoch_metrics)
            self._send_metrics_update()
        
        # Estimate remaining time
        if self.timing_stats['avg_epoch_time'] > 0:
            remaining_epochs = self.current_progress.total_epochs - self.current_progress.current_epoch
            estimated_remaining = remaining_epochs * self.timing_stats['avg_epoch_time']
            self.current_progress.metrics['estimated_remaining_time'] = estimated_remaining
    
    def complete_training(self, final_metrics: Dict[str, float] = None,
                         best_checkpoint: str = None) -> None:
        """Mark completion of training.
        
        Args:
            final_metrics: Final training metrics
            best_checkpoint: Path to best checkpoint
        """
        self.current_progress.overall_progress = 100.0
        self.current_progress.epoch_progress = 100.0
        self.current_progress.batch_progress = 100.0
        self.current_progress.phase = "completed"
        self.current_progress.message = "✅ Training completed!"
        
        # Calculate total time
        if self.pipeline_start_time:
            total_time = time.time() - self.pipeline_start_time
            self.timing_stats['total_time'] = total_time
            self.current_progress.metrics['total_training_time'] = total_time
        
        # Add final metrics
        if final_metrics:
            self.current_progress.metrics.update(final_metrics)
        
        if best_checkpoint:
            self.current_progress.metrics['best_checkpoint'] = best_checkpoint
        
        # Send final updates
        self._send_progress_update()
        self._send_metrics_update()
        
        # Complete UI progress tracker
        self._update_ui_progress_tracker("complete", "Training completed successfully!")
    
    def training_error(self, error_message: str, exception: Exception = None) -> None:
        """Handle training error.
        
        Args:
            error_message: Error message for display
            exception: Exception object (optional)
        """
        self.current_progress.phase = "error"
        self.current_progress.message = f"❌ Error: {error_message}"
        
        if exception:
            self.current_progress.metrics['error_type'] = type(exception).__name__
            self.current_progress.metrics['error_details'] = str(exception)
        
        # Send error update
        self._send_progress_update()
        
        # Update UI progress tracker
        self._update_ui_progress_tracker("error", error_message)
    
    # Generic operation methods (from ModelProgressBridge)
    def start_operation(self, operation_name: str, total_steps: int) -> None:
        """Start operation with total steps."""
        self.current_operation = operation_name
        self.operation_steps = total_steps
        self.current_step = 0
        self.substep_current = 0
        self.substep_total = 0
        
        logger.info(f"🚀 Starting: {operation_name} ({total_steps} steps)")
        self._notify_progress(0, total_steps, f"🚀 Starting {operation_name}...", "overall")
    
    def update_operation(self, step: int, message: str, phase: str = "current") -> None:
        """Update progress for main operation steps."""
        self.current_step = step
        
        progress_percentage = (step / max(self.operation_steps, 1)) * 100
        logger.debug(f"📊 {self.current_operation}: {progress_percentage:.1f}% - {message}")
        
        self._notify_progress(step, self.operation_steps, message, phase)
    
    def update_substep(self, substep: int, substep_total: int, message: str, phase: str = "current") -> None:
        """Update progress for substeps within main step."""
        self.substep_current = substep
        self.substep_total = substep_total
        
        # Calculate combined progress: main step + substep progress
        main_progress = self.current_step
        substep_progress = (substep / max(substep_total, 1)) * 0.8  # 80% of current step
        combined_current = main_progress + substep_progress
        
        logger.debug(f"📋 Substep {substep}/{substep_total}: {message}")
        self._notify_progress(int(combined_current * 10), self.operation_steps * 10, message, phase)
    
    def complete_operation(self, final_step: int, message: str) -> None:
        """Mark operation as complete."""
        self.current_step = final_step
        
        logger.info(f"✅ {self.current_operation} completed: {message}")
        self._notify_progress(final_step, self.operation_steps, message, "overall")
        
        # Reset operation state
        self._reset_operation_state()
    
    def operation_error(self, error_message: str, phase: str = "current") -> None:
        """Report error in operation."""
        logger.error(f"❌ {self.current_operation} error: {error_message}")
        
        # Notify error through callback if supported
        if self.progress_callback:
            try:
                # Try to call error method if progress tracker supports it
                if hasattr(self.progress_callback, 'error'):
                    self.progress_callback.error(error_message, phase)
                elif callable(self.progress_callback):
                    # Fallback: call as regular progress with error indicator
                    self.progress_callback("error", 0, 1, f"❌ {error_message}")
            except Exception as e:
                logger.warning(f"⚠️ Error calling progress callback: {str(e)}")
        
        self._reset_operation_state()
    
    def set_callback(self, callback: Callable) -> None:
        """Set or update progress callback."""
        self.progress_callback = callback
        logger.debug(f"🔄 Progress callback updated: {type(callback).__name__ if hasattr(callback, '__name__') else 'callable'}")
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status."""
        if not self.current_operation:
            return {"status": "idle", "message": "No operation in progress"}
        
        main_progress = (self.current_step / max(self.operation_steps, 1)) * 100
        substep_progress = (self.substep_current / max(self.substep_total, 1)) * 100 if self.substep_total > 0 else 0
        
        return {
            "status": "active",
            "operation": self.current_operation,
            "main_progress": main_progress,
            "current_step": self.current_step,
            "total_steps": self.operation_steps,
            "substep_progress": substep_progress,
            "substep_current": self.substep_current,
            "substep_total": self.substep_total
        }
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get timing statistics summary."""
        return {
            'total_time': self.timing_stats['total_time'],
            'avg_epoch_time': self.timing_stats['avg_epoch_time'],
            'completed_epochs': len(self.timing_stats['epoch_times']),
            'estimated_total_time': self.timing_stats['avg_epoch_time'] * self.current_progress.total_epochs if self.timing_stats['avg_epoch_time'] > 0 else 0
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current progress state."""
        return {
            'progress': {
                'overall': self.current_progress.overall_progress,
                'epoch': self.current_progress.epoch_progress,
                'batch': self.current_progress.batch_progress
            },
            'current': {
                'epoch': self.current_progress.current_epoch,
                'batch': self.current_progress.current_batch,
                'phase': self.current_progress.phase
            },
            'total': {
                'epochs': self.current_progress.total_epochs,
                'batches': self.current_progress.total_batches
            },
            'message': self.current_progress.message,
            'metrics': self.current_progress.metrics.copy(),
            'timing': self.get_timing_summary()
        }
    
    # Private helper methods
    def _send_progress_update(self) -> None:
        """Send progress update to callbacks."""
        if self.progress_callback:
            try:
                # Format data for callback
                progress_data = {
                    'overall_progress': self.current_progress.overall_progress,
                    'epoch_progress': self.current_progress.epoch_progress,
                    'batch_progress': self.current_progress.batch_progress,
                    'current_epoch': self.current_progress.current_epoch,
                    'total_epochs': self.current_progress.total_epochs,
                    'current_batch': self.current_progress.current_batch,
                    'total_batches': self.current_progress.total_batches,
                    'phase': self.current_progress.phase,
                    'message': self.current_progress.message,
                    'metrics': self.current_progress.metrics.copy()
                }
                
                self.progress_callback(progress_data)
            except Exception:
                # Silent fail to avoid disrupting training
                pass
        
        # Update UI progress tracker
        self._update_ui_tracker_progress()
    
    def _send_metrics_update(self) -> None:
        """Send metrics update to callback."""
        if self.metrics_callback and self.current_progress.metrics:
            try:
                metrics_data = {
                    'epoch': self.current_progress.current_epoch,
                    'phase': self.current_progress.phase,
                    'metrics': self.current_progress.metrics.copy(),
                    'timing': self.timing_stats.copy()
                }
                
                self.metrics_callback(metrics_data)
            except Exception:
                # Silent fail
                pass
    
    def _update_ui_progress_tracker(self, action: str, message: str = "", steps: List[str] = None) -> None:
        """Update UI Progress Tracker if available."""
        tracker = self.ui_components.get('progress_tracker')
        if not tracker:
            return
        
        try:
            if action == "show":
                tracker.show(message, steps or [])
            elif action == "complete":
                tracker.complete(message)
            elif action == "error":
                tracker.error(message)
        except Exception:
            # Silent fail
            pass
    
    def _update_ui_tracker_progress(self) -> None:
        """Update progress bars in UI tracker."""
        tracker = self.ui_components.get('progress_tracker')
        if not tracker:
            return
        
        try:
            # Update overall progress
            tracker.update_overall(
                self.current_progress.overall_progress,
                f"Epoch {self.current_progress.current_epoch}/{self.current_progress.total_epochs}"
            )
            
            # Update current progress (batch level)
            if self.current_progress.total_batches > 0:
                tracker.update_current(
                    self.current_progress.batch_progress,
                    f"Batch {self.current_progress.current_batch}/{self.current_progress.total_batches}"
                )
        except Exception:
            # Silent fail
            pass
    
    def _notify_progress(self, current: int, total: int, message: str, phase: str) -> None:
        """Notify progress to callback with error handling."""
        if not self.progress_callback:
            return
        
        try:
            # Support different callback formats
            if hasattr(self.progress_callback, 'update'):
                # Progress Tracker object with update method
                self.progress_callback.update(current, total, message, phase)
            elif callable(self.progress_callback):
                # Function callback
                self.progress_callback(phase, current, total, message)
            else:
                logger.warning("⚠️ Invalid progress callback format")
                
        except Exception as e:
            logger.error(f"⚠️ Error calling progress callback: {str(e)}")
            import traceback
            logger.debug(f"Progress callback error traceback: {traceback.format_exc()}")
    
    def _reset_operation_state(self) -> None:
        """Reset internal operation state."""
        self.current_operation = ""
        self.operation_steps = 0
        self.current_step = 0
        self.substep_current = 0
        self.substep_total = 0
    
    # Backward compatibility methods
    def complete(self, step: int, message: str) -> None:
        """Backward compatibility alias for complete_operation."""
        self.complete_operation(step, message)
    
    def error(self, error_message: str) -> None:
        """Backward compatibility alias for operation_error."""
        self.operation_error(error_message)


class ProgressContext:
    """Context manager for automatic progress tracking."""
    
    def __init__(self, tracker: TrainingProgressTracker, operation_name: str, total_steps: int):
        self.tracker = tracker
        self.operation_name = operation_name
        self.total_steps = total_steps
    
    def __enter__(self):
        self.tracker.start_operation(self.operation_name, self.total_steps)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.tracker.complete_operation(self.total_steps, f"✅ {self.operation_name} completed successfully")
        else:
            self.tracker.operation_error(f"❌ {self.operation_name} failed: {str(exc_val)}")


# Convenience functions and factory methods
def create_training_progress_tracker(progress_callback: Optional[Callable] = None,
                                    verbose: bool = True,
                                    training_mode: str = 'two_phase',
                                    ui_components: Optional[Dict[str, Any]] = None,
                                    metrics_callback: Optional[Callable] = None) -> TrainingProgressTracker:
    """Factory function to create TrainingProgressTracker."""
    return TrainingProgressTracker(progress_callback, verbose, training_mode, ui_components, metrics_callback)

def create_training_progress_bridge(ui_components: Dict[str, Any] = None,
                                   progress_callback: Callable = None,
                                   metrics_callback: Callable = None) -> TrainingProgressTracker:
    """Factory function for training progress bridge (backward compatibility)."""
    return TrainingProgressTracker(progress_callback, True, 'two_phase', ui_components, metrics_callback)

def create_progress_bridge(callback: Optional[Callable] = None) -> TrainingProgressTracker:
    """Factory function for model progress bridge (backward compatibility)."""
    return TrainingProgressTracker(callback, True, 'single_phase')

def create_simple_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create simple progress callback for basic UI updates."""
    def progress_callback(progress_data: Dict[str, Any]) -> None:
        message = progress_data.get('message', 'Processing...')
        overall = progress_data.get('overall_progress', 0)
        
        # Log to UI if logger available
        logger_obj = ui_components.get('logger')
        if logger_obj:
            logger_obj.info(f"📊 {message} ({overall:.1f}%)")
    
    return progress_callback

def progress_context(tracker: TrainingProgressTracker, operation: str, steps: int) -> ProgressContext:
    """Create progress context manager."""
    return ProgressContext(tracker, operation, steps)