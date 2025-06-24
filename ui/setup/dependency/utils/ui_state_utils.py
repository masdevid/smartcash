# File: smartcash/ui/setup/dependency/utils/ui_state_utils.py
# Deskripsi: UI state management utilities tanpa circular import

import functools
from typing import Dict, Any, Optional, Callable, List

def clear_ui_outputs(ui_components: Dict[str, Any], output_keys: Optional[list] = None):
    """Clear multiple UI outputs dengan one-liner pattern"""
    output_keys = output_keys or ['log_output', 'status', 'confirmation_area']
    [widget.clear_output(wait=True) for key in output_keys 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan consolidated approach - one-liner"""
    ui_components.get('status_panel') and (
        lambda panel: __import__('smartcash.ui.components', fromlist=['update_status_panel']).update_status_panel(panel, message, status_type)
    )(ui_components['status_panel'])

def log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Safe logging ke UI output dengan error handling - one-liner"""
    log_output = ui_components.get('log_output')
    log_output and hasattr(log_output, 'append_stdout') and (
        setattr(log_output, '_content', getattr(log_output, '_content', []) + 
               [{"name": "stdout", "output_type": "stream", "text": f"[{level.upper()}] {message}\n"}]) or
        None
    )

def reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger output untuk clear previous logs - one-liner"""
    log_output = ui_components.get('log_output')
    log_output and hasattr(log_output, 'clear_output') and log_output.clear_output(wait=True)

class ProgressSteps:
    """Progress step constants untuk konsistensi"""
    INIT = "init"
    ANALYSIS = "analysis" 
    INSTALLATION = "installation"
    COMPLETE = "complete"
    ERROR = "error"

def update_progress_step(ui_components: Dict[str, Any], progress_type: str, value: int, 
                         message: str = "", color: str = None):
    """Update progress dengan API progress tracker baru dan safe error handling"""
    try:
        # Gunakan fungsi update_progress dari ui_components jika tersedia
        update_progress = ui_components.get('update_progress')
        if update_progress and callable(update_progress):
            update_progress(progress_type, value, message, color)
            return
        
        # Fallback ke progress_tracker langsung
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if progress_type == 'overall' or progress_type == 'level1':
                hasattr(progress_tracker, 'update_overall') and progress_tracker.update_overall(value, message)
            elif progress_type == 'step' or progress_type == 'level2':
                hasattr(progress_tracker, 'update_step') and progress_tracker.update_step(value, message)
                
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Progress update error: {str(e)}", "warning")

def show_progress_tracker_safe(ui_components: Dict[str, Any], operation: str = "", steps: List[str] = None):
    """Safely show progress tracker dengan error handling"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        progress_tracker and hasattr(progress_tracker, 'show') and progress_tracker.show(
            operation=operation or "🔄 Memproses...",
            steps=steps or ["Mempersiapkan", "Memproses", "Menyelesaikan"]
        )
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Progress tracker show error: {str(e)}", "warning")

def reset_progress_tracker_safe(ui_components: Dict[str, Any]):
    """Safely reset progress tracker dengan error handling"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        progress_tracker and hasattr(progress_tracker, 'reset') and progress_tracker.reset()
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Progress tracker reset error: {str(e)}", "warning")

def create_operation_context(ui_components: Dict[str, Any], operation_name: str):
    """Create operation context untuk tracking operasi dengan progress"""
    
    class OperationContext:
        def __init__(self, ui_components: Dict[str, Any], operation_name: str):
            self.ui_components = ui_components
            self.operation_name = operation_name
            
        def __enter__(self):
            log_to_ui_safe(self.ui_components, f"🚀 Memulai {self.operation_name}...", "info")
            show_progress_tracker_safe(self.ui_components, f"🔄 {self.operation_name}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                complete_operation_with_message(self.ui_components, f"✅ {self.operation_name} berhasil")
            else:
                error_operation_with_message(self.ui_components, f"❌ {self.operation_name} gagal: {str(exc_val)}")
            reset_progress_tracker_safe(self.ui_components)
    
    return OperationContext(ui_components, operation_name)

def complete_operation_with_message(ui_components: Dict[str, Any], message: str):
    """Complete operation dengan success message"""
    update_progress_step(ui_components, 'overall', 100, message, 'green')
    log_to_ui_safe(ui_components, message, "success")
    update_status_panel(ui_components, message, "success")

def error_operation_with_message(ui_components: Dict[str, Any], message: str):
    """Complete operation dengan error message"""
    update_progress_step(ui_components, 'overall', 100, message, 'red')
    log_to_ui_safe(ui_components, message, "error")
    update_status_panel(ui_components, message, "error")

def with_button_context(ui_components: Dict[str, Any], operation: str = "operation"):
    """Decorator untuk menangani state tombol secara otomatis"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Daftar tombol yang akan di-disable
            buttons = [ui_components.get(btn) for btn in 
                      ['install_button', 'analyze_button', 'save_button', 'reset_button', 'cleanup_button']
                      if ui_components.get(btn)]
            
            # Nonaktifkan semua tombol
            for btn in buttons:
                hasattr(btn, 'disabled') and setattr(btn, 'disabled', True)
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                log_to_ui_safe(ui_components, f"⚠️ Error in {operation}: {str(e)}", "error")
                raise
            finally:
                # Aktifkan kembali semua tombol
                for btn in buttons:
                    hasattr(btn, 'disabled') and setattr(btn, 'disabled', False)
        
        return wrapper
    return decorator

def create_progress_tracker(ui_components: Dict[str, Any]) -> Callable:
    """Create progress tracking function dengan closure - one-liner factory"""
    def progress_tracker_wrapper(progress_type, value, message="", color=None):
        update_progress_step(ui_components, progress_type, value, message, color)
    return progress_tracker_wrapper

def create_logger_bridge(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Create logger bridge untuk kompatibilitas dengan logger interface"""
    return {
        'info': lambda msg: log_to_ui_safe(ui_components, f"ℹ️ {msg}", "info"),
        'warning': lambda msg: log_to_ui_safe(ui_components, f"⚠️ {msg}", "warning"),
        'error': lambda msg: log_to_ui_safe(ui_components, f"❌ {msg}", "error"),
        'success': lambda msg: log_to_ui_safe(ui_components, f"✅ {msg}", "success"),
        'debug': lambda msg: log_to_ui_safe(ui_components, f"🔍 {msg}", "debug")
    }