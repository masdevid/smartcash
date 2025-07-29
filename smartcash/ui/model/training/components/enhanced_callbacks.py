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
            try:
                # Update UI progress if available
                if self.ui_module and hasattr(self.ui_module, '_update_progress_display'):
                    try:
                        progress_data = {
                            'type': progress_type,
                            'current': current,
                            'total': total,
                            'message': message,
                            'percentage': (current / total * 100) if total > 0 else 0,
                            **kwargs
                        }
                        self.ui_module._update_progress_display(progress_data)
                    except Exception as e:
                        print(f"UI progress update failed: {e}")
                
                # Memory monitoring for batch progress
                if progress_type == 'batch' and self.verbose and current % 10 == 0:
                    self._monitor_memory(current, kwargs.get('epoch', 0))\n                \n                # Console output\n                if self.verbose:\n                    percentage = (current / total) * 100 if total > 0 else 0\n                    display_name = self._format_progress_type(progress_type)\n                    print(f"🔄 PROGRESS [{display_name}] {percentage:.0f}% ({current}/{total}): {message}")\n                \n            except Exception as e:\n                print(f"❌ Progress callback error: {e}")\n        \n        return progress_callback\n    \n    def create_live_chart_callback(self) -> Callable:\n        """Create live chart callback for real-time visualization."""\n        def live_chart_callback(epoch: int, phase: int, metrics: Dict[str, Any], layer_metrics: Dict[str, Any]):\n            """Handle live chart updates."""\n            try:\n                # Filter metrics for current phase\n                filtered_metrics = filter_phase_relevant_metrics(metrics, phase)\n                \n                # Prepare chart data\n                chart_data = {\n                    'epoch': epoch,\n                    'phase': phase,\n                    'metrics': filtered_metrics,\n                    'layer_metrics': layer_metrics,\n                    'timestamp': time.time()\n                }\n                \n                # Update UI charts if available\n                if self.ui_module and hasattr(self.ui_module, '_update_live_charts'):\n                    try:\n                        self.ui_module._update_live_charts(chart_data)\n                    except Exception as e:\n                        print(f"UI chart update failed: {e}")\n                \n                # Console summary for verbose mode\n                if self.verbose:\n                    key_metrics = self._get_key_metrics_summary(filtered_metrics)\n                    print(f"📊 EPOCH {epoch} SUMMARY: {key_metrics}")\n                    \n            except Exception as e:\n                print(f"❌ Live chart callback error: {e}")\n        \n        return live_chart_callback\n    \n    def _handle_ui_metrics_update(self, phase: str, epoch: int, metrics: Dict[str, Any], colored_metrics: Dict[str, Dict]):\n        """Handle UI-specific metrics updates."""\n        try:\n            if self.ui_module and hasattr(self.ui_module, '_handle_colored_metrics'):\n                self.ui_module._handle_colored_metrics(phase, epoch, metrics, colored_metrics)\n        except Exception as e:\n            print(f"UI metrics handler error: {e}")\n    \n    def _update_training_state(self, phase: str, epoch: int, metrics: Dict[str, Any]):\n        """Update internal training state based on phase information."""\n        # Determine current phase from phase string\n        if 'phase_1' in phase.lower() or phase.lower() == 'training_phase_1':\n            self.training_state['current_phase'] = 1\n        elif 'phase_2' in phase.lower() or phase.lower() == 'training_phase_2':\n            self.training_state['current_phase'] = 2\n        elif 'single' in phase.lower():\n            self.training_state['training_mode'] = 'single_phase'\n            self.training_state['current_phase'] = 1\n        \n        self.training_state['epoch'] = epoch\n    \n    def _monitor_memory(self, batch: int, epoch: int):\n        """Monitor memory usage during training."""\n        try:\n            current_memory = self._get_memory_usage()\n            \n            # Check for significant memory increase\n            if self.last_memory_check > 0 and current_memory - self.last_memory_check > 500:\n                # Perform light cleanup\n                if torch.cuda.is_available():\n                    torch.cuda.empty_cache()\n                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n                    try:\n                        torch.mps.empty_cache()\n                    except Exception:\n                        pass\n                \n                print(f"🧹 Memory cleanup performed: {current_memory:.0f}MB")\n            \n            self.last_memory_check = current_memory\n            \n        except Exception:\n            pass  # Ignore memory monitoring errors\n    \n    def _get_memory_usage(self) -> float:\n        """Get current memory usage in MB."""\n        try:\n            process = psutil.Process()\n            return process.memory_info().rss / 1024 / 1024\n        except Exception:\n            return 0.0\n    \n    def _format_progress_type(self, progress_type: str) -> str:\n        """Format progress type for display."""\n        type_mapping = {\n            'overall': 'Overall Training',\n            'epoch': 'Epoch Progress',\n            'batch': 'Batch Progress'\n        }\n        return type_mapping.get(progress_type, progress_type.replace('_', ' ').title())\n    \n    def _get_key_metrics_summary(self, metrics: Dict[str, Any]) -> str:\n        """Get summary of key metrics for console output."""\n        key_metrics = []\n        \n        if 'train_loss' in metrics:\n            key_metrics.append(f"Train Loss: {metrics['train_loss']:.4f}")\n        if 'val_loss' in metrics:\n            key_metrics.append(f"Val Loss: {metrics['val_loss']:.4f}")\n        if 'layer_1_accuracy' in metrics:\n            key_metrics.append(f"L1 Acc: {metrics['layer_1_accuracy']:.3f}")\n        if 'val_map50' in metrics:\n            key_metrics.append(f"mAP@0.5: {metrics['val_map50']:.3f}")\n        \n        return " | ".join(key_metrics) if key_metrics else "No key metrics available"\n    \n    def cleanup_memory(self, verbose: bool = None):\n        """Comprehensive memory cleanup."""\n        if verbose is None:\n            verbose = self.verbose\n            \n        if verbose:\n            memory_before = self._get_memory_usage()\n            print(f"🧹 MEMORY CLEANUP: Starting cleanup (Memory: {memory_before:.1f} MB)")\n        \n        # Clear PyTorch caches\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n            torch.cuda.synchronize()\n            if verbose:\n                print("   ✅ Cleared CUDA cache")\n        \n        # Clear MPS cache if available (Apple Silicon)\n        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n            try:\n                torch.mps.empty_cache()\n                if verbose:\n                    print("   ✅ Cleared MPS cache")\n            except Exception:\n                pass\n        \n        # Force garbage collection\n        collected = gc.collect()\n        if verbose:\n            print(f"   ✅ Garbage collection: {collected} objects collected")\n        \n        # Additional aggressive cleanup\n        for _ in range(3):\n            gc.collect()\n        \n        if verbose:\n            memory_after = self._get_memory_usage()\n            memory_freed = memory_before - memory_after\n            print(f"   ✅ Memory cleanup complete (Memory: {memory_after:.1f} MB, Freed: {memory_freed:.1f} MB)")\n\n\ndef create_enhanced_training_callbacks(ui_module=None, verbose: bool = True) -> Dict[str, Callable]:\n    """Factory function to create all enhanced training callbacks.\n    \n    Args:\n        ui_module: Reference to UI module for integration\n        verbose: Enable verbose output\n        \n    Returns:\n        Dictionary of callback functions\n    """\n    callbacks = EnhancedTrainingCallbacks(ui_module, verbose)\n    \n    return {\n        'log_callback': callbacks.create_log_callback(),\n        'metrics_callback': callbacks.create_metrics_callback(),\n        'progress_callback': callbacks.create_progress_callback(),\n        'live_chart_callback': callbacks.create_live_chart_callback(),\n        'cleanup_function': callbacks.cleanup_memory\n    }\n\n\n# Backward compatibility\ndef create_training_callbacks(ui_module=None, verbose: bool = True) -> Dict[str, Callable]:\n    """Backward compatibility wrapper."""\n    return create_enhanced_training_callbacks(ui_module, verbose)