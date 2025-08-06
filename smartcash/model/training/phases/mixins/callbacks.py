"""
Callbacks mixin for training phase management.

This module provides a mixin class that encapsulates all callback functionality
including log_callback, metrics_callback, live_chart_callback, and progress_callback.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class CallbacksMixin:
    """Mixin class that provides callback functionality for training components."""
    
    def __init__(self, *args, **kwargs):
        """Initialize callbacks mixin."""
        super().__init__(*args, **kwargs)
        
        # Initialize callback references
        self._log_callback: Optional[Callable] = None
        self._metrics_callback: Optional[Callable] = None
        self._live_chart_callback: Optional[Callable] = None
        self._progress_callback: Optional[Callable] = None
    
    def set_callbacks(self, log_callback: Optional[Callable] = None,
                     metrics_callback: Optional[Callable] = None,
                     live_chart_callback: Optional[Callable] = None,
                     progress_callback: Optional[Callable] = None):
        """
        Set all callback functions at once.
        
        Args:
            log_callback: Function to handle log messages
            metrics_callback: Function to handle metrics updates
            live_chart_callback: Function to handle live chart updates
            progress_callback: Function to handle progress updates
        """
        self._log_callback = log_callback
        self._metrics_callback = metrics_callback
        self._live_chart_callback = live_chart_callback
        self._progress_callback = progress_callback
        
        logger.debug(f"Set callbacks: log={log_callback is not None}, "
                    f"metrics={metrics_callback is not None}, "
                    f"live_chart={live_chart_callback is not None}, "
                    f"progress={progress_callback is not None}")
    
    def set_log_callback(self, callback: Optional[Callable]):
        """Set log callback function."""
        self._log_callback = callback
        
    def set_metrics_callback(self, callback: Optional[Callable]):
        """Set metrics callback function."""
        self._metrics_callback = callback
        
    def set_live_chart_callback(self, callback: Optional[Callable]):
        """Set live chart callback function."""
        self._live_chart_callback = callback
        
    def set_progress_callback(self, callback: Optional[Callable]):
        """Set progress callback function."""
        self._progress_callback = callback
    
    def emit_log(self, level: str, message: str, data: Dict[str, Any] = None):
        """
        Emit log message through log callback.
        
        Args:
            level: Log level ('info', 'warning', 'error', 'debug', 'critical')
            message: Log message
            data: Optional additional data context
        """
        if self._log_callback:
            try:
                self._log_callback(level, message, data or {})
            except Exception as e:
                logger.warning(f"Log callback failed: {e}")
    
    def emit_metrics(self, phase_name: str, epoch: int, metrics: Dict[str, Any], **kwargs):
        """
        Emit metrics through metrics callback.
        
        Args:
            phase_name: Phase name (e.g., 'phase_1', 'training_phase_single')
            epoch: Current epoch number
            metrics: Metrics dictionary
            **kwargs: Additional keyword arguments (e.g., loss_breakdown)
        """
        if self._metrics_callback:
            try:
                # Support multiple callback signatures for backward compatibility
                try:
                    self._metrics_callback(phase_name, epoch, metrics, **kwargs)
                except TypeError:
                    # Fallback for older callback signatures
                    try:
                        self._metrics_callback(phase_name, epoch, metrics)
                    except TypeError:
                        self._metrics_callback(phase_name, metrics)
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")
    
    def emit_live_chart(self, level: str, message: str, data: Dict[str, Any] = None):
        """
        Emit live chart update through live chart callback.
        
        Args:
            level: Chart level ('info', 'success', 'warning', 'error')  
            message: Chart message
            data: Optional chart data
        """
        if self._live_chart_callback:
            try:
                # Support multiple callback signatures
                try:
                    self._live_chart_callback(level, message, data or {})
                except TypeError:
                    # Fallback for older callback signatures
                    self._live_chart_callback(level, data or {})
            except Exception as e:
                logger.warning(f"Live chart callback failed: {e}")
    
    def emit_progress(self, progress_type: str, current: int, total: int, 
                     message: str = "", **kwargs):
        """
        Emit progress update through progress callback.
        
        Args:
            progress_type: Type of progress ('overall', 'epoch', 'batch', 'phase_1', etc.)
            current: Current progress value
            total: Total progress value
            message: Optional progress message
            **kwargs: Additional progress context (e.g., phase, epoch)
        """
        if self._progress_callback:
            try:
                self._progress_callback(progress_type, current, total, message, **kwargs)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def emit_epoch_updates(self, epoch: int, phase_num: int, metrics: Dict[str, Any]):
        """
        Emit epoch updates using both metrics and live chart callbacks.
        
        Args:
            epoch: Current epoch number
            phase_num: Current phase number
            metrics: Epoch metrics dictionary
        """
        phase_str = f'phase_{phase_num}'
        
        # Emit metrics update
        self.emit_metrics(phase_str, epoch, metrics)
        
        # Emit live chart update
        self.emit_live_chart('info', f"Epoch {epoch} metrics updated", metrics)
        
        # Emit log message
        val_acc = metrics.get('val_accuracy', 0)
        val_loss = metrics.get('val_loss', 0)
        self.emit_log('info', 
                     f"Epoch {epoch}/{phase_num} - Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}",
                     {'phase': phase_num, 'epoch': epoch})
    
    def emit_phase_completion(self, phase_num: int, best_metrics: Dict[str, Any], 
                             final_metrics: Dict[str, Any]):
        """
        Emit phase completion notifications.
        
        Args:
            phase_num: Completed phase number
            best_metrics: Best metrics achieved in this phase
            final_metrics: Final epoch metrics
        """
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
        
        # Emit completion notifications
        self.emit_metrics(phase_str, current_epoch, completion_metrics)
        self.emit_live_chart('success', f"Phase {phase_num} completed", completion_metrics)
        self.emit_log('info', f"✅ Phase {phase_num} training completed", 
                     {'phase': phase_num, 'final_epoch': current_epoch})
    
    def emit_training_start(self, phase_num: int, total_epochs: int, resume_epoch: int = 0):
        """
        Emit training start notifications.
        
        Args:
            phase_num: Starting phase number
            total_epochs: Total epochs for this phase
            resume_epoch: Resume epoch (0 for fresh start)
        """
        if resume_epoch > 0:
            message = f"🔄 Resuming Phase {phase_num} training from epoch {resume_epoch+1}/{total_epochs}"
            data = {'phase': phase_num, 'resumed_from': resume_epoch, 'total_epochs': total_epochs}
        else:
            message = f"🚀 Starting Phase {phase_num} training for {total_epochs} epochs"
            data = {'phase': phase_num, 'total_epochs': total_epochs}
        
        self.emit_log('info', message, data)
        self.emit_live_chart('info', f"Phase {phase_num} training started", data)
    
    def emit_training_error(self, phase_num: int, error_message: str, error_data: Dict[str, Any] = None):
        """
        Emit training error notifications.
        
        Args:
            phase_num: Phase number where error occurred
            error_message: Error message
            error_data: Optional error context data
        """
        data = {'phase': phase_num, **(error_data or {})}
        
        self.emit_log('error', f"❌ Phase {phase_num} training error: {error_message}", data)
        self.emit_live_chart('error', f"Phase {phase_num} training failed", data)
    
    def emit_best_model_saved(self, phase_num: int, epoch: int, metrics: Dict[str, Any], 
                             checkpoint_path: str):
        """
        Emit best model saved notifications.
        
        Args:
            phase_num: Phase number
            epoch: Epoch when best model was achieved
            metrics: Best model metrics
            checkpoint_path: Path where checkpoint was saved
        """
        data = {
            'phase': phase_num, 
            'epoch': epoch, 
            'checkpoint_path': checkpoint_path,
            **metrics
        }
        
        val_acc = metrics.get('val_accuracy', 0)
        message = f"💾 New best model saved: Phase {phase_num}, Epoch {epoch}, Acc: {val_acc:.4f}"
        
        self.emit_log('info', message, data)
        self.emit_live_chart('success', "New best model saved", data)
    
    def has_callbacks(self) -> bool:
        """Check if any callbacks are set."""
        return any([
            self._log_callback is not None,
            self._metrics_callback is not None,
            self._live_chart_callback is not None,
            self._progress_callback is not None
        ])
    
    def cleanup_callbacks(self):
        """Clean up callback references."""
        self._log_callback = None
        self._metrics_callback = None
        self._live_chart_callback = None
        self._progress_callback = None
    
    @property
    def log_callback(self) -> Optional[Callable]:
        """Get log callback reference."""
        return self._log_callback
    
    @property
    def metrics_callback(self) -> Optional[Callable]:
        """Get metrics callback reference."""
        return self._metrics_callback
    
    @property
    def live_chart_callback(self) -> Optional[Callable]:
        """Get live chart callback reference."""
        return self._live_chart_callback
    
    @property
    def progress_callback(self) -> Optional[Callable]:
        """Get progress callback reference."""
        return self._progress_callback