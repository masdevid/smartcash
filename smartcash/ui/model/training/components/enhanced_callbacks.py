"""
File: smartcash/ui/model/training/components/enhanced_callbacks.py
Description: Enhanced callback implementations based on latest backend improvements.
"""

import gc
import time
import psutil
import torch
from typing import Dict, Any, Optional, Callable, List
from smartcash.model.training.utils.metric_color_utils import ColorScheme
from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback
from smartcash.model.training.utils.metrics_utils import filter_phase_relevant_metrics
from .phase_aware_metrics import generate_phase_aware_metrics_html, create_live_metrics_update


class EnhancedTrainingCallbacks:
    """Enhanced training callbacks with latest backend improvements and phase-aware metrics."""
    
    def __init__(self, ui_module=None, verbose: bool = True):
        """Initialize enhanced callbacks.
        
        Args:
            ui_module: Reference to the UI module for updates
            verbose: Enable verbose output
        """
        self.ui_module = ui_module
        self.verbose = verbose
        self.last_memory_check = 0
        self.training_state = {
            'current_phase': 1,
            'training_mode': 'two_phase',
            'epoch': 0,
            'total_epochs': 0
        }
        
        # Create UI-enhanced metrics callback
        self.metrics_callback_impl = create_ui_metrics_callback(
            verbose=verbose,
            console_scheme=ColorScheme.EMOJI,
            ui_callback=self._handle_ui_metrics_update
        )
    
    def create_log_callback(self) -> Callable:
        """Create enhanced log callback with UI integration."""
        def log_callback(level: str, message: str, data: dict = None):
            """Handle log messages from the training pipeline."""
            # Format level with appropriate emoji
            level_icons = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌',
                'debug': '🔍',
                'critical': '🚨'
            }
            
            icon = level_icons.get(level.lower(), '📝')
            formatted_message = f"{icon} [{level.upper()}] {message}"
            
            # Send to UI if available
            if self.ui_module and hasattr(self.ui_module, '_update_log_output'):
                try:
                    self.ui_module._update_log_output(formatted_message)
                except Exception as e:
                    print(f"UI log update failed: {e}")
            
            # Console output
            if self.verbose:
                print(formatted_message)
                
                # Print additional data if available
                if data:
                    for key, value in data.items():
                        if key != 'message':  # Avoid duplicate message
                            print(f"    {key}: {value}")
        
        return log_callback
    
    def create_metrics_callback(self) -> Callable:
        """Create enhanced metrics callback with phase-aware UI updates."""
        def metrics_callback(phase: str, epoch: int, metrics: Dict[str, Any], **kwargs):
            """Handle metrics from training pipeline."""
            try:
                # Update training state
                self._update_training_state(phase, epoch, metrics)
                
                # Get phase-aware metrics display
                current_phase = self.training_state['current_phase']
                training_mode = self.training_state['training_mode']
                
                # Create live update data
                update_data = create_live_metrics_update(
                    metrics, current_phase, epoch, training_mode
                )
                
                # Update UI metrics display
                if self.ui_module and hasattr(self.ui_module, '_update_metrics_display'):
                    try:
                        self.ui_module._update_metrics_display(update_data)
                    except Exception as e:
                        print(f"UI metrics update failed: {e}")
                
                # Console output via UI callback
                result = self.metrics_callback_impl(phase, epoch, update_data['metrics'], **kwargs)
                return result
                
            except Exception as e:
                print(f"❌ Metrics callback error: {e}")
                return {'error': str(e)}
        
        return metrics_callback
    
    def create_progress_callback(self) -> Callable:
        """Create enhanced progress callback with 3-level system and memory monitoring."""
        def progress_callback(progress_type: str, current: int, total: int, message: str = "", **kwargs):
            """Handle progress updates with 3-level system."""
            # Update UI progress if available
            if self.ui_module and hasattr(self.ui_module, '_update_progress_display'):
                progress_data = {
                    'type': progress_type,
                    'current': current,
                    'total': total,
                    'message': message,
                    'percentage': (current / total * 100) if total > 0 else 0,
                    **kwargs
                    }
                self.ui_module._update_progress_display(progress_data)