"""
File: smartcash/dataset/augmentor/communicator.py
Deskripsi: Fixed UI communication bridge dengan log method yang benar dan one-liner optimizations
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

class UICommunicator:
    """Fixed unified interface untuk UI communication dengan log method yang lengkap."""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        """Initialize UI communicator dengan components dan logger bridge."""
        self.ui_components = ui_components or {}
        self.logger = self._setup_logger()
        self.progress_tracker = self._get_progress_tracker()
        
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
        """Dapatkan progress tracker dari UI components."""
        return self.ui_components.get('tracker')
    
    # One-liner progress method
    progress = lambda self, op, curr, total, msg="": (
        self.progress_tracker.update(op, min(100, max(0, int((curr / max(1, total)) * 100))), msg)
        if self.progress_tracker and hasattr(self.progress_tracker, 'update') else None
    )
    
    # One-liner log methods dengan fallback yang aman
    log_info = lambda self, msg: getattr(self.logger, 'info', lambda x: print(f"ℹ️ {x}"))(f"ℹ️ {msg}")
    log_success = lambda self, msg: getattr(self.logger, 'success', lambda x: print(f"✅ {x}"))(f"✅ {msg}")
    log_warning = lambda self, msg: getattr(self.logger, 'warning', lambda x: print(f"⚠️ {x}"))(f"⚠️ {msg}")
    log_error = lambda self, msg: getattr(self.logger, 'error', lambda x: print(f"❌ {x}"))(f"❌ {msg}")
    log_debug = lambda self, msg: getattr(self.logger, 'debug', lambda x: print(f"🔍 {x}"))(f"🔍 {msg}")
    
    # Generic log method untuk backward compatibility
    def log(self, level: str, message: str):
        """Generic log method dengan level mapping."""
        log_methods = {
            'info': self.log_info,
            'success': self.log_success,
            'warning': self.log_warning,
            'error': self.log_error,
            'debug': self.log_debug
        }
        log_method = log_methods.get(level, self.log_info)
        log_method(message)
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Mulai operasi dengan progress tracking."""
        self.log_info(f"🚀 Memulai {operation_name}")
        if self.progress_tracker and hasattr(self.progress_tracker, 'show'):
            try:
                self.progress_tracker.show(operation_name.lower().replace(' ', '_'))
            except Exception:
                pass
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """Selesaikan operasi dengan progress tracking."""
        final_message = result_message or f"{operation_name} selesai"
        self.log_success(final_message)
        
        if self.progress_tracker and hasattr(self.progress_tracker, 'complete'):
            try:
                self.progress_tracker.complete(final_message)
            except Exception:
                pass
    
    def error_operation(self, operation_name: str, error_message: str):
        """Handle error operasi dengan progress tracking."""
        final_message = f"{operation_name} gagal: {error_message}"
        self.log_error(final_message)
        
        if self.progress_tracker and hasattr(self.progress_tracker, 'error'):
            try:
                self.progress_tracker.error(final_message)
            except Exception:
                pass
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status UI dengan pesan."""
        self.log(status_type, message)
        
        # Update status panel jika tersedia
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    # One-liner helper methods
    is_stop_requested = lambda self: self.ui_components.get('stop_requested', False)
    
    def report_progress_with_callback(self, progress_callback: Optional[Callable] = None, 
                                    step: str = "overall", current: int = 0, 
                                    total: int = 100, message: str = ""):
        """Report progress dengan callback dan tracker sekaligus."""
        self.progress(step, current, total, message)
        
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(step, current, total, message)
            except Exception:
                pass

def create_communicator(ui_components: Dict[str, Any] = None) -> UICommunicator:
    """Factory function untuk membuat UI communicator."""
    return UICommunicator(ui_components)

# One-liner helper functions
log_to_ui = lambda comm, msg, level="info": comm.log(level, msg)
progress_to_ui = lambda comm, op, curr, total, msg="": comm.progress(op, curr, total, msg)
start_ui_operation = lambda comm, name: comm.start_operation(name)
complete_ui_operation = lambda comm, name, msg="": comm.complete_operation(name, msg)
error_ui_operation = lambda comm, name, msg: comm.error_operation(name, msg)