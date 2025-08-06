"""
Progress tracking mixin for phase management.

Handles progress tracking, metrics emission, and callback management.
"""

import inspect
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class ProgressTrackingMixin:
    """Mixin for progress tracking and callback management."""
    
    def emit_epoch_updates(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """Emit epoch updates through callbacks."""
        # Ensure phase is a string and properly formatted
        phase_str = f'phase_{phase_num}'
        
        # Get loss breakdown if available
        loss_breakdown = final_metrics.get('loss_breakdown', {})
        
        # Call metrics callback with all required parameters if available
        if hasattr(self, 'emit_metrics_callback') and self.emit_metrics_callback:
            try:
                # First try with full parameter set
                if callable(getattr(self.emit_metrics_callback, '__code__', None)):
                    # Count number of parameters the callback expects
                    param_count = self.emit_metrics_callback.__code__.co_argcount
                    
                    if param_count >= 4:  # phase, epoch, metrics, **kwargs
                        self.emit_metrics_callback(
                            phase=phase_str,
                            epoch=epoch,
                            metrics=final_metrics,
                            loss_breakdown=loss_breakdown
                        )
                    elif param_count == 3:  # phase, epoch, metrics
                        self.emit_metrics_callback(phase_str, epoch, final_metrics)
                    elif param_count == 2:  # epoch, metrics
                        self.emit_metrics_callback(epoch, final_metrics)
                    else:
                        # Fallback to minimal call
                        self.emit_metrics_callback(epoch, final_metrics)
                else:
                    # Fallback to minimal call if we can't inspect the callback
                    self.emit_metrics_callback(epoch, final_metrics)
                        
            except Exception as e:
                logger = get_logger(self.__class__.__name__)
                logger.warning(f"Error in metrics callback: {str(e)}")
                # Fallback to minimal call
                try:
                    self.emit_metrics_callback(epoch, final_metrics)
                except Exception as inner_e:
                    logger.error(f"Fallback metrics callback failed: {str(inner_e)}")
        
        # Handle live chart callback if available
        if hasattr(self, 'emit_live_chart_callback') and self.emit_live_chart_callback:
            try:
                if callable(getattr(self.emit_live_chart_callback, '__code__', None)):
                    # Count number of parameters the callback expects
                    param_count = self.emit_live_chart_callback.__code__.co_argcount
                    
                    if param_count >= 4:  # epoch, phase_num, metrics, layer_metrics
                        self.emit_live_chart_callback(epoch, phase_num, final_metrics, layer_metrics)
                    elif param_count == 2:  # epoch, metrics
                        self.emit_live_chart_callback(epoch, final_metrics)
                    else:
                        # Fallback to minimal call
                        self.emit_live_chart_callback(epoch, final_metrics)
                else:
                    # Fallback to minimal call if we can't inspect the callback
                    self.emit_live_chart_callback(epoch, final_metrics)
                        
            except Exception as e:
                logger = get_logger(self.__class__.__name__)
                logger.warning(f"Error in live chart callback: {str(e)}")
                # Fallback to minimal call
                try:
                    self.emit_live_chart_callback(epoch, final_metrics)
                except Exception as inner_e:
                    logger.error(f"Fallback live chart callback failed: {str(inner_e)}")
    
    def handle_early_stop_cleanup(self, epoch: int, final_metrics: dict, phase_num: int,
                                early_stopping, progress_manager):
        """Handle cleanup when early stopping is triggered."""
        logger = get_logger(self.__class__.__name__)
        
        if early_stopping and early_stopping.should_stop:
            # Create early stopping message
            stop_info = early_stopping.get_best_info() if hasattr(early_stopping, 'get_best_info') else {}
            stop_reason = stop_info.get('stop_reason', 'Early stopping criteria met')
            
            early_stop_message = f"Early stopping at epoch {epoch + 1}: {stop_reason}"
            
            # Update progress with early stopping
            if progress_manager:
                progress_manager.complete_epoch_early_stopping(epoch, early_stop_message)
            
            logger.info(f"🛑 {early_stop_message}")
            
            # Add early stopping info to metrics
            final_metrics['early_stopped'] = True
            final_metrics['early_stop_epoch'] = epoch + 1
            final_metrics['early_stop_reason'] = stop_reason
    
    def update_epoch_completion_progress(self, epoch: int, epochs: int, 
                                       progress_message: str, progress_manager,
                                       custom_percentage: Optional[float] = None):
        """Update progress tracking for epoch completion."""
        if progress_manager:
            progress_manager.update_epoch_progress(
                epoch, epochs, progress_message, custom_percentage
            )