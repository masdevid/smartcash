"""
File: smartcash/dataset/augmentor/communicator.py
Deskripsi: Fixed communicator dengan aggressive log suppression dan UI-only output
"""

from typing import Dict, Any, Optional, Callable
import time
import logging
import sys

class UICommunicator:
    """Fixed communicator dengan aggressive suppression dan UI-only output"""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        self.ui_components = ui_components or {}
        self.progress_tracker = self._get_progress_tracker()
        self.last_update_time = 0
        self.update_interval = 0.5  # Throttling interval
        
        # Setup aggressive suppression IMMEDIATELY
        self._setup_aggressive_suppression()
        
        # Setup minimal logger yang HANYA ke UI
        self.logger = self._setup_ui_only_logger()
        
    def _setup_aggressive_suppression(self):
        """Setup aggressive log suppression untuk prevent console leakage"""
        # Suppress EVERYTHING di console
        console_suppressions = [
            '', 'root', 'smartcash', 'smartcash.dataset', 'smartcash.dataset.augmentor',
            'smartcash.common', 'albumentations', 'cv2', 'numpy', 'PIL', 'matplotlib',
            'concurrent.futures', 'threading', 'requests', 'urllib3', 'ipywidgets'
        ]
        
        for target in console_suppressions:
            try:
                logger = logging.getLogger(target)
                logger.setLevel(logging.CRITICAL + 1)  # Above CRITICAL
                logger.propagate = False
                # Remove ALL handlers
                logger.handlers.clear()
                # Add null handler
                logger.addHandler(logging.NullHandler())
            except Exception:
                pass
        
        # Suppress root logger completely
        try:
            root = logging.getLogger()
            root.setLevel(logging.CRITICAL + 1)
            root.handlers.clear()
            root.addHandler(logging.NullHandler())
        except Exception:
            pass
        
        # Redirect stdout/stderr jika perlu
        self._suppress_stdout_stderr()
    
    def _suppress_stdout_stderr(self):
        """Suppress stdout/stderr untuk prevent accidental prints"""
        if not hasattr(self.ui_components, '_stdout_suppressed'):
            try:
                # Create null output
                import os
                null_device = open(os.devnull, 'w')
                
                # Store original untuk restoration
                self.ui_components['_original_stdout'] = sys.stdout
                self.ui_components['_original_stderr'] = sys.stderr
                
                # Suppress hanya jika bukan IPython environment
                if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
                    sys.stdout = null_device
                    sys.stderr = null_device
                
                self.ui_components['_stdout_suppressed'] = True
            except Exception:
                pass
    
    def _setup_ui_only_logger(self):
        """Setup logger yang HANYA output ke UI"""
        # Buat dummy logger yang tidak pernah print ke console
        class UIOnlyLogger:
            def info(self, msg): pass
            def success(self, msg): pass
            def warning(self, msg): pass
            def error(self, msg): pass
            def debug(self, msg): pass
        
        return UIOnlyLogger()
    
    def _get_progress_tracker(self):
        """Get progress tracker dengan fallback methods"""
        return self.ui_components.get('tracker') or self.ui_components
    
    def progress(self, operation: str, current: int, total: int, message: str = ""):
        """Fixed progress dengan UI-only updates dan throttling"""
        current_time = time.time()
        
        # Throttle updates
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        if (current_time - self.last_update_time < self.update_interval and 
            percentage not in [0, 100]):
            return
        
        self.last_update_time = current_time
        
        # Update HANYA ke UI progress tracker
        try:
            if hasattr(self.progress_tracker, 'update'):
                self.progress_tracker.update(operation, percentage, message)
            elif 'update_progress' in self.progress_tracker:
                self.progress_tracker['update_progress'](operation, percentage, message)
        except Exception:
            pass  # Silent fail - TIDAK print error
    
    def log_info(self, msg: str):
        """Log info HANYA ke UI"""
        self._log_to_ui_only(f"ℹ️ {msg}", 'info')
    
    def log_success(self, msg: str):
        """Log success HANYA ke UI"""
        self._log_to_ui_only(f"✅ {msg}", 'success')
    
    def log_warning(self, msg: str):
        """Log warning HANYA ke UI"""
        self._log_to_ui_only(f"⚠️ {msg}", 'warning')
    
    def log_error(self, msg: str):
        """Log error HANYA ke UI"""
        self._log_to_ui_only(f"❌ {msg}", 'error')
    
    def log_debug(self, msg: str):
        """Log debug HANYA ke UI"""
        self._log_to_ui_only(f"🔍 {msg}", 'debug')
    
    def _log_to_ui_only(self, message: str, level: str):
        """Log HANYA ke UI output - TIDAK ke console"""
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
            
            # Display HANYA ke UI log widgets
            for output_key in ['log_output', 'status', 'output']:
                output_widget = self.ui_components.get(output_key)
                if output_widget and hasattr(output_widget, 'clear_output'):
                    with output_widget:
                        display(HTML(html_msg))
                    break
        except Exception:
            pass  # Silent fail - TIDAK print ke console
    
    def log(self, level: str, message: str):
        """Generic log method dengan UI-only output"""
        log_methods = {
            'info': self.log_info, 'success': self.log_success, 'warning': self.log_warning,
            'error': self.log_error, 'debug': self.log_debug
        }
        log_method = log_methods.get(level, self.log_info)
        log_method(message)
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Start operation dengan UI-only feedback"""
        self.log_info(f"🚀 Memulai {operation_name}")
        self.last_update_time = 0  # Reset throttling
        
        try:
            if hasattr(self.progress_tracker, 'show'):
                self.progress_tracker.show(operation_name.lower().replace(' ', '_'))
            elif 'show_for_operation' in self.progress_tracker:
                self.progress_tracker['show_for_operation'](operation_name.lower().replace(' ', '_'))
        except Exception:
            pass
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """Complete operation dengan UI-only feedback"""
        final_message = result_message or f"{operation_name} selesai"
        self.log_success(final_message)
        
        try:
            if hasattr(self.progress_tracker, 'complete'):
                self.progress_tracker.complete(final_message)
            elif 'complete_operation' in self.progress_tracker:
                self.progress_tracker['complete_operation'](final_message)
        except Exception:
            pass
    
    def error_operation(self, operation_name: str, error_message: str):
        """Handle error dengan UI-only feedback"""
        final_message = f"{operation_name} gagal: {error_message}"
        self.log_error(final_message)
        
        try:
            if hasattr(self.progress_tracker, 'error'):
                self.progress_tracker.error(final_message)
            elif 'error_operation' in self.progress_tracker:
                self.progress_tracker['error_operation'](final_message)
        except Exception:
            pass
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status dengan UI-only output"""
        self.log(status_type, message)
    
    def report_progress_with_callback(self, progress_callback: Optional[Callable] = None, 
                                    step: str = "overall", current: int = 0, 
                                    total: int = 100, message: str = ""):
        """Enhanced progress reporting dengan dual tracking"""
        # Update our progress
        self.progress(step, current, total, message)
        
        # Call external callback dengan error suppression
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(step, current, total, message)
            except Exception:
                pass  # Silent fail
    
    def cleanup_suppression(self):
        """Cleanup suppression jika diperlukan"""
        try:
            if self.ui_components.get('_stdout_suppressed'):
                # Restore original stdout/stderr
                if '_original_stdout' in self.ui_components:
                    sys.stdout = self.ui_components['_original_stdout']
                if '_original_stderr' in self.ui_components:
                    sys.stderr = self.ui_components['_original_stderr']
                
                self.ui_components['_stdout_suppressed'] = False
        except Exception:
            pass
    
    # One-liner helper methods
    is_stop_requested = lambda self: self.ui_components.get('stop_requested', False)

def create_communicator(ui_components: Dict[str, Any] = None) -> UICommunicator:
    """Factory function untuk membuat UI communicator dengan aggressive suppression"""
    return UICommunicator(ui_components)

# One-liner helper functions dengan suppression
log_to_ui = lambda comm, msg, level="info": comm.log(level, msg) if comm else None
progress_to_ui = lambda comm, op, curr, total, msg="": comm.progress(op, curr, total, msg) if comm else None
start_ui_operation = lambda comm, name: comm.start_operation(name) if comm else None
complete_ui_operation = lambda comm, name, msg="": comm.complete_operation(name, msg) if comm else None
error_ui_operation = lambda comm, name, msg: comm.error_operation(name, msg) if comm else None