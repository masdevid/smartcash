"""
File: smartcash/dataset/augmentor/communicator.py
Deskripsi: Fixed UI communication bridge dengan real-time progress updates dan proper log streaming
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger
import time

class UICommunicator:
    """Fixed unified interface untuk UI communication dengan real-time progress streaming."""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        """Initialize UI communicator dengan components dan real-time tracker."""
        self.ui_components = ui_components or {}
        self.logger = self._setup_logger()
        self.progress_tracker = self._get_progress_tracker()
        self.last_update_time = 0
        self.update_interval = 0.5  # Minimum interval between updates (seconds)
        
    def _setup_logger(self):
        """Setup logger dengan UI bridge jika tersedia."""
        if not self.ui_components:
            return get_logger("augmentation")
            
        try:
            from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
            return create_ui_logger_bridge(self.ui_components, "augmentation")
        except ImportError:
            return get_logger("augmentation")
    
    def _get_progress_tracker(self):
        """Dapatkan progress tracker dari UI components dengan fallback methods."""
        return self.ui_components.get('tracker') or self.ui_components
    
    def progress(self, operation: str, current: int, total: int, message: str = ""):
        """Fixed progress dengan real-time updates dan throttling."""
        current_time = time.time()
        
        # Throttle updates untuk prevent flooding (kecuali 0% atau 100%)
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        if (current_time - self.last_update_time < self.update_interval and 
            percentage not in [0, 100]):
            return
        
        self.last_update_time = current_time
        
        # Update progress tracker dengan multiple methods
        if hasattr(self.progress_tracker, 'update'):
            self.progress_tracker.update(operation, percentage, message)
        elif 'update_progress' in self.progress_tracker:
            self.progress_tracker['update_progress'](operation, percentage, message)
        
        # Also log to output untuk visibility
        if message and percentage % 10 == 0:  # Log every 10%
            self.log_info(f"📊 {operation.title()}: {percentage}% - {message}")
    
    def log_info(self, msg: str):
        """Fixed info logging dengan immediate UI display."""
        formatted_msg = f"ℹ️ {msg}"
        self._log_to_output(formatted_msg, 'info')
        getattr(self.logger, 'info', lambda x: None)(formatted_msg)
    
    def log_success(self, msg: str):
        """Fixed success logging dengan immediate UI display."""
        formatted_msg = f"✅ {msg}"
        self._log_to_output(formatted_msg, 'success')
        getattr(self.logger, 'success', lambda x: None)(formatted_msg)
    
    def log_warning(self, msg: str):
        """Fixed warning logging dengan immediate UI display."""
        formatted_msg = f"⚠️ {msg}"
        self._log_to_output(formatted_msg, 'warning')
        getattr(self.logger, 'warning', lambda x: None)(formatted_msg)
    
    def log_error(self, msg: str):
        """Fixed error logging dengan immediate UI display."""
        formatted_msg = f"❌ {msg}"
        self._log_to_output(formatted_msg, 'error')
        getattr(self.logger, 'error', lambda x: None)(formatted_msg)
    
    def log_debug(self, msg: str):
        """Fixed debug logging dengan immediate UI display."""
        formatted_msg = f"🔍 {msg}"
        self._log_to_output(formatted_msg, 'debug')
        getattr(self.logger, 'debug', lambda x: None)(formatted_msg)
    
    def _log_to_output(self, message: str, level: str):
        """Immediate log ke UI output dengan color coding."""
        try:
            from IPython.display import display, HTML
            
            # Color mapping
            colors = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545', 'debug': '#6c757d'}
            color = colors.get(level, '#007bff')
            
            # Timestamp untuk real-time feeling
            timestamp = time.strftime('%H:%M:%S')
            
            html_msg = f"""
            <div style="margin:1px 0;padding:2px 8px;border-radius:3px;
                       font-family:'Courier New',monospace;font-size:12px;
                       border-left:3px solid {color};">
                <span style="color:#6c757d;font-size:10px;">[{timestamp}]</span> 
                <span style="color:{color};">{message}</span>
            </div>
            """
            
            # Display ke log_output dengan priority order
            for output_key in ['log_output', 'status', 'output']:
                output_widget = self.ui_components.get(output_key)
                if output_widget and hasattr(output_widget, 'clear_output'):
                    with output_widget:
                        display(HTML(html_msg))
                    break
        except Exception:
            # Silent fallback
            pass
    
    def log(self, level: str, message: str):
        """Generic log method dengan level mapping."""
        log_methods = {
            'info': self.log_info, 'success': self.log_success, 'warning': self.log_warning,
            'error': self.log_error, 'debug': self.log_debug
        }
        log_method = log_methods.get(level, self.log_info)
        log_method(message)
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Mulai operasi dengan progress tracking yang enhanced."""
        self.log_info(f"🚀 Memulai {operation_name}")
        self.last_update_time = 0  # Reset throttling
        
        if hasattr(self.progress_tracker, 'show'):
            self.progress_tracker.show(operation_name.lower().replace(' ', '_'))
        elif 'show_for_operation' in self.progress_tracker:
            self.progress_tracker['show_for_operation'](operation_name.lower().replace(' ', '_'))
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """Selesaikan operasi dengan progress tracking."""
        final_message = result_message or f"{operation_name} selesai"
        self.log_success(final_message)
        
        if hasattr(self.progress_tracker, 'complete'):
            self.progress_tracker.complete(final_message)
        elif 'complete_operation' in self.progress_tracker:
            self.progress_tracker['complete_operation'](final_message)
    
    def error_operation(self, operation_name: str, error_message: str):
        """Handle error operasi dengan progress tracking."""
        final_message = f"{operation_name} gagal: {error_message}"
        self.log_error(final_message)
        
        if hasattr(self.progress_tracker, 'error'):
            self.progress_tracker.error(final_message)
        elif 'error_operation' in self.progress_tracker:
            self.progress_tracker['error_operation'](final_message)
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status UI dengan pesan real-time."""
        self.log(status_type, message)
        
        # Update status panel jika tersedia
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def report_progress_with_callback(self, progress_callback: Optional[Callable] = None, 
                                    step: str = "overall", current: int = 0, 
                                    total: int = 100, message: str = ""):
        """Enhanced progress reporting dengan dual tracking."""
        # Update our progress
        self.progress(step, current, total, message)
        
        # Also call external callback
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(step, current, total, message)
            except Exception:
                pass
    
    # One-liner helper methods
    is_stop_requested = lambda self: self.ui_components.get('stop_requested', False)

def create_communicator(ui_components: Dict[str, Any] = None) -> UICommunicator:
    """Factory function untuk membuat UI communicator dengan real-time updates."""
    return UICommunicator(ui_components)

# One-liner helper functions
log_to_ui = lambda comm, msg, level="info": comm.log(level, msg)
progress_to_ui = lambda comm, op, curr, total, msg="": comm.progress(op, curr, total, msg)
start_ui_operation = lambda comm, name: comm.start_operation(name)
complete_ui_operation = lambda comm, name, msg="": comm.complete_operation(name, msg)
error_ui_operation = lambda comm, name, msg: comm.error_operation(name, msg)