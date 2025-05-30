"""
File: smartcash/dataset/augmentor/communicator.py
Deskripsi: Fixed communicator dengan selective log suppression tanpa auto-reset
"""

from typing import Dict, Any, Optional, Callable
import time
import logging

class UICommunicator:
    """Fixed communicator dengan selective suppression tanpa auto log reset"""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        self.ui_components = ui_components or {}
        self.progress_tracker = self._get_progress_tracker()
        self.last_update_time = 0
        self.update_interval = 0.3
        self.current_operation = None
        
        # Setup aggressive suppression untuk verbose libraries
        self._setup_selective_suppression()
        
    def _setup_selective_suppression(self):
        """Setup selective suppression - suppress verbose libraries aggressively"""
        # Aggressive suppression untuk libraries yang sangat verbose
        verbose_loggers = [
            'requests', 'urllib3', 'http.client', 'requests.packages.urllib3',
            'albumentations', 'cv2', 'numpy', 'PIL', 'matplotlib', 
            'concurrent.futures', 'threading', 'multiprocessing',
            'tqdm', 'ipywidgets', 'traitlets', 'tornado', 'zmq'
        ]
        
        for logger_name in verbose_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1)  # Above CRITICAL
            logger.propagate = False
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
    
    def _get_progress_tracker(self):
        """Get progress tracker dengan fallback methods"""
        return self.ui_components.get('tracker') or self.ui_components
    
    def progress(self, operation: str, current: int, total: int, message: str = ""):
        """Progress dengan proper completion tracking"""
        current_time = time.time()
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        
        # Skip throttling untuk milestone updates
        is_milestone = percentage in [0, 25, 50, 75, 100] or (current_time - self.last_update_time > self.update_interval)
        
        if not is_milestone and percentage not in [0, 100]:
            return
        
        self.last_update_time = current_time
        
        # Update UI tracker
        try:
            if hasattr(self.progress_tracker, 'update'):
                self.progress_tracker.update(operation, percentage, message)
            elif 'update_progress' in self.progress_tracker:
                self.progress_tracker['update_progress'](operation, percentage, message)
        except Exception:
            pass
        
        # Log milestone tanpa reset
        if percentage in [0, 25, 50, 75, 100]:
            self._log_progress_milestone(operation, percentage, message)
    
    def _log_progress_milestone(self, operation: str, percentage: int, message: str):
        """Log milestone dengan emoji mapping"""
        milestone_emoji = {0: "🚀", 25: "📊", 50: "⚡", 75: "🎯", 100: "✅"}
        emoji = milestone_emoji.get(percentage, "📈")
        
        milestone_msg = f"{emoji} {operation.title()}: {percentage}% - {message}"
        self._log_to_ui_clean(milestone_msg, 'info')
    
    def log_info(self, msg: str):
        """Log info tanpa reset"""
        self._log_to_ui_clean(f"ℹ️ {msg}", 'info')
    
    def log_success(self, msg: str):
        """Log success tanpa reset"""
        self._log_to_ui_clean(f"✅ {msg}", 'success')
    
    def log_warning(self, msg: str):
        """Log warning tanpa reset"""
        self._log_to_ui_clean(f"⚠️ {msg}", 'warning')
    
    def log_error(self, msg: str):
        """Log error tanpa reset"""
        self._log_to_ui_clean(f"❌ {msg}", 'error')
    
    def _log_to_ui_clean(self, message: str, level: str):
        """Log ke UI tanpa auto reset"""
        try:
            from IPython.display import display, HTML
            import time
            
            colors = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = colors.get(level, '#007bff')
            
            timestamp = time.strftime('%H:%M:%S')
            
            html_msg = f"""
            <div style="margin:2px 0;padding:4px 8px;border-radius:3px;
                       font-family:'Courier New',monospace;font-size:13px;
                       border-left:3px solid {color};">
                <span style="color:#6c757d;font-size:11px;">[{timestamp}]</span> 
                <span style="color:{color};font-weight:500;">{message}</span>
            </div>
            """
            
            for output_key in ['log_output', 'status', 'output']:
                output_widget = self.ui_components.get(output_key)
                if output_widget and hasattr(output_widget, 'clear_output'):
                    with output_widget:
                        display(HTML(html_msg))
                    break
        except Exception:
            print(message)
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Start operation tanpa log reset"""
        self.current_operation = operation_name
        self.log_info(f"🚀 Memulai {operation_name}")
        self.last_update_time = 0
        
        try:
            if hasattr(self.progress_tracker, 'show'):
                self.progress_tracker.show(operation_name.lower().replace(' ', '_'))
            elif 'show_for_operation' in self.progress_tracker:
                self.progress_tracker['show_for_operation'](operation_name.lower().replace(' ', '_'))
        except Exception:
            pass
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """Complete operation dengan proper 100% progress"""
        final_message = result_message or f"{operation_name} selesai"
        
        # Ensure 100% progress
        self.progress("overall", 100, 100, "Completed")
        
        self.log_success(final_message)
        
        try:
            if hasattr(self.progress_tracker, 'complete'):
                self.progress_tracker.complete(final_message)
            elif 'complete_operation' in self.progress_tracker:
                self.progress_tracker['complete_operation'](final_message)
        except Exception:
            pass
        
        self.current_operation = None
    
    def error_operation(self, operation_name: str, error_message: str):
        """Handle error dengan clear feedback"""
        final_message = f"{operation_name} gagal: {error_message}"
        self.log_error(final_message)
        
        try:
            if hasattr(self.progress_tracker, 'error'):
                self.progress_tracker.error(final_message)
            elif 'error_operation' in self.progress_tracker:
                self.progress_tracker['error_operation'](final_message)
        except Exception:
            pass
        
        self.current_operation = None
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status tanpa reset"""
        status_methods = {'info': self.log_info, 'success': self.log_success, 'warning': self.log_warning, 'error': self.log_error}
        log_method = status_methods.get(status_type, self.log_info)
        log_method(message)
    
    def report_progress_with_callback(self, progress_callback: Optional[Callable] = None, 
                                    step: str = "overall", current: int = 0, 
                                    total: int = 100, message: str = ""):
        """Report progress dengan dual tracking"""
        self.progress(step, current, total, message)
        
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(step, current, total, message)
            except Exception:
                pass

def create_communicator(ui_components: Dict[str, Any] = None) -> UICommunicator:
    """Factory function untuk communicator dengan selective suppression"""
    return UICommunicator(ui_components)

# One-liner helpers
log_to_ui = lambda comm, msg, level="info": comm.update_status(msg, level) if comm else print(f"{level}: {msg}")
progress_to_ui = lambda comm, op, curr, total, msg="": comm.progress(op, curr, total, msg) if comm else print(f"{op}: {curr}/{total}")
start_ui_operation = lambda comm, name: comm.start_operation(name) if comm else print(f"START: {name}")
complete_ui_operation = lambda comm, name, msg="": comm.complete_operation(name, msg) if comm else print(f"COMPLETE: {name}")
error_ui_operation = lambda comm, name, msg: comm.error_operation(name, msg) if comm else print(f"ERROR: {name}")