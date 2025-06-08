"""
File: smartcash/ui/dataset/augmentation/utils/progress_utils.py
Deskripsi: Unified progress tracking yang menggantikan backend progress tracker
"""

from typing import Dict, Any, Callable, Optional
import time

class UnifiedProgressManager:
    """Unified progress manager yang menggantikan backend progress tracker"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.last_update = {'time': 0, 'percentage': -1, 'step': ''}
        self.current_operation = None
        self.operation_start_time = None
    
    def create_backend_communicator(self):
        """Create communicator interface untuk backend service"""
        return BackendProgressCommunicator(self)
    
    def create_progress_callback(self) -> Callable:
        """Create progress callback untuk service integration"""
        def callback(step: str, current: int, total: int, message: str):
            self._update_progress_with_throttling(step, current, total, message)
        return callback
    
    def start_operation(self, operation_name: str):
        """Start operation tracking"""
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        self._show_progress_container()
        self._log_ui(f"🚀 Memulai {operation_name}", 'info')
    
    def complete_operation(self, message: str, success: bool = True):
        """Complete operation dengan summary"""
        if self.operation_start_time:
            duration = time.time() - self.operation_start_time
            status_emoji = "✅" if success else "❌"
            final_msg = f"{status_emoji} {message} (⏱️ {duration:.1f}s)"
        else:
            final_msg = f"{'✅' if success else '❌'} {message}"
        
        self._update_progress_tracker('overall', 100, final_msg)
        self._log_ui(final_msg, 'success' if success else 'error')
        self.current_operation = None
    
    def error_operation(self, error_message: str):
        """Handle operation error"""
        self.complete_operation(f"Error: {error_message}", success=False)
    
    def _update_progress_with_throttling(self, step: str, current: int, total: int, message: str):
        """Update progress dengan throttling untuk performance"""
        current_time = time.time()
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        
        # Throttling logic - update pada milestone atau interval
        should_update = (
            step != self.last_update['step'] or
            percentage in [0, 25, 50, 75, 100] or
            (current_time - self.last_update['time'] > 0.5)
        )
        
        if should_update:
            self.last_update.update({
                'time': current_time, 
                'percentage': percentage, 
                'step': step
            })
            
            # Update UI tracker
            self._update_progress_tracker(step, percentage, message)
            
            # Log milestone progress
            if percentage in [0, 25, 50, 75, 100]:
                step_emoji = {'overall': '🎯', 'step': '📊', 'current': '⚡'}.get(step, '📈')
                progress_msg = f"{step_emoji} {step.title()}: {percentage}% - {message}"
                self._log_ui(progress_msg, 'info')
    
    def _update_progress_tracker(self, level: str, percentage: int, message: str):
        """Update progress tracker dengan safe method calls"""
        try:
            progress_tracker = self.ui_components.get('progress_tracker')
            if not progress_tracker:
                return
            
            # Map level ke method yang sesuai
            if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(percentage, message)
            elif level == 'step' and hasattr(progress_tracker, 'update_step'):
                progress_tracker.update_step(percentage, message)
            elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                progress_tracker.update_current(percentage, message)
            elif hasattr(progress_tracker, 'update'):
                progress_tracker.update(level, percentage, message)
        except Exception:
            pass
    
    def _show_progress_container(self):
        """Show progress container untuk operation"""
        try:
            # Show progress tracker jika ada method show
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'show'):
                progress_tracker.show()
            
            # Alternative: show via function reference
            show_func = self.ui_components.get('show_for_operation')
            if show_func and callable(show_func):
                show_func(self.current_operation)
        except Exception:
            pass
    
    def _log_ui(self, message: str, level: str = 'info'):
        """Log ke UI dengan fallback chain"""
        try:
            # Priority 1: UI Logger
            logger = self.ui_components.get('logger')
            if logger and hasattr(logger, level):
                getattr(logger, level)(message)
                return
            
            # Priority 2: Log widget
            widget = self.ui_components.get('log_output') or self.ui_components.get('status')
            if widget and hasattr(widget, 'clear_output'):
                from IPython.display import display, HTML
                color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
                color = color_map.get(level, '#007bff')
                html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
                
                with widget:
                    display(HTML(html))
                return
        except Exception:
            pass
        
        # Fallback print
        print(message)

class BackendProgressCommunicator:
    """Communicator interface untuk backend service compatibility"""
    
    def __init__(self, progress_manager: UnifiedProgressManager):
        self.progress_manager = progress_manager
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Interface untuk backend start operation"""
        self.progress_manager.start_operation(operation_name)
    
    def progress(self, step: str, current: int, total: int, message: str = ""):
        """Interface untuk backend progress updates"""
        self.progress_manager._update_progress_with_throttling(step, current, total, message)
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """Interface untuk backend complete operation"""
        self.progress_manager.complete_operation(result_message or f"{operation_name} selesai")
    
    def error_operation(self, operation_name: str, error_message: str):
        """Interface untuk backend error operation"""
        self.progress_manager.error_operation(f"{operation_name}: {error_message}")
    
    def log_info(self, message: str):
        """Interface untuk backend logging"""
        self.progress_manager._log_ui(message, 'info')
    
    def log_success(self, message: str):
        """Interface untuk backend logging"""
        self.progress_manager._log_ui(message, 'success')
    
    def log_warning(self, message: str):
        """Interface untuk backend logging"""
        self.progress_manager._log_ui(message, 'warning')
    
    def log_error(self, message: str):
        """Interface untuk backend logging"""
        self.progress_manager._log_ui(message, 'error')

# Factory functions
def create_unified_progress_manager(ui_components: Dict[str, Any]) -> UnifiedProgressManager:
    """Factory untuk unified progress manager"""
    return UnifiedProgressManager(ui_components)

def create_backend_communicator(ui_components: Dict[str, Any]) -> BackendProgressCommunicator:
    """Factory untuk backend communicator interface"""
    manager = create_unified_progress_manager(ui_components)
    return manager.create_backend_communicator()

# Backward compatibility
create_progress_manager = create_unified_progress_manager