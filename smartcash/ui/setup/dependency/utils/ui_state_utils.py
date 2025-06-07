"""
File: smartcash/ui/setup/dependency/utils/ui_state_utils.py
Deskripsi: UI state management utilities dengan integrasi progress tracker baru
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.setup.dependency.components.package_selector import update_package_status
from smartcash.ui.components.progress_tracker import ProgressTracker

def clear_ui_outputs(ui_components: Dict[str, Any], output_keys: Optional[list] = None):
    """Clear multiple UI outputs dengan one-liner pattern"""
    output_keys = output_keys or ['log_output', 'status', 'confirmation_area']
    [widget.clear_output(wait=True) for key in output_keys 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan consolidated approach - one-liner"""
    ui_components.get('status_panel') and (
        lambda panel: __import__('smartcash.ui.components.status_panel', fromlist=['update_status_panel']).update_status_panel(panel, message, status_type)
    )(ui_components['status_panel'])

def reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger output untuk clear previous logs - one-liner"""
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)

def update_progress_step(ui_components: Dict[str, Any], progress_type: str, value: int, 
                         message: str = "", color: str = None):
    """Update progress dengan API progress tracker baru dan safe error handling"""
    try:
        # Gunakan fungsi update_progress dari ui_components jika tersedia (backward compatibility wrapper)
        update_progress = ui_components.get('update_progress')
        if update_progress and callable(update_progress):
            update_progress(progress_type, value, message, color)
            return
        
        # Jika tidak ada update_progress, gunakan progress_tracker langsung
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Gunakan metode yang benar sesuai API progress tracker baru
            if progress_type == 'overall' or progress_type == 'level1':
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(value, message, color)
            elif progress_type == 'step' or progress_type == 'level2':
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(value, message, color)
            elif progress_type == 'step_progress' or progress_type == 'level3':
                # Untuk triple level progress tracker
                if hasattr(progress_tracker, 'update_step_progress'):
                    progress_tracker.update_step_progress(value, message, color)
                elif hasattr(progress_tracker, 'update_current'):
                    # Fallback ke current jika tidak ada step_progress
                    progress_tracker.update_current(value, message, color)
    except Exception as e:
        # Silent fail untuk compatibility
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"🔄 Progress update error (non-critical): {str(e)}")
        pass

def show_operation_progress(ui_components: Dict[str, Any], operation: str, 
                           steps: List[str] = None, weights: Dict[str, int] = None):
    """Show progress container untuk operation dengan steps dan weights kustom"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Definisi steps default berdasarkan jenis operasi
        operation_steps = {
            'install': ["🔧 Persiapan", "🔍 Analisis", "📦 Instalasi", "✅ Finalisasi"],
            'analyze': ["🔧 Persiapan", "🔍 Scanning", "📊 Evaluasi", "📝 Laporan"],
            'status': ["🔧 Persiapan", "💻 Sistem Info", "📦 Package Check", "📝 Laporan"]
        }
        
        # Gunakan steps yang diberikan atau default berdasarkan operasi
        steps_to_use = steps or operation_steps.get(operation, ["🔧 Persiapan", "⚙️ Proses", "✅ Finalisasi"])
        
        # Tampilkan progress tracker dengan steps dan weights yang sesuai
        progress_tracker.show(f"{operation.title()}", steps_to_use, weights)
    else:
        ui_components.get('show_for_operation', lambda x: None)(operation)

def show_progress_tracker_safe(ui_components: Dict[str, Any], title: str, 
                              steps: List[str], weights: Dict[str, int] = None):
    """Show progress tracker dengan safe fallback dan dukungan weights"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Gunakan API show dengan weights jika tersedia
        progress_tracker.show(title, steps, weights)
    else:
        ui_components.get('show_for_operation', lambda x: None)(title)

def reset_progress_tracker_safe(ui_components: Dict[str, Any]):
    """Reset progress tracker dengan safe fallback dan reset semua level"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Reset semua level progress tracker
        progress_tracker.reset()
        
        # Reset nilai progress di semua level ke 0
        progress_tracker.update_overall(0, "")
        if hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(0, "")
        if hasattr(progress_tracker, 'update_step_progress'):
            progress_tracker.update_step_progress(0, "")
    else:
        # Tidak ada fallback untuk reset dalam API lama
        pass

def complete_operation_with_message(ui_components: Dict[str, Any], message: str, delay: float = 1.0):
    """Complete operation dengan success message dan opsional delay"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Update semua level ke 100% sebelum menampilkan pesan complete
        progress_tracker.update_overall(100, "✅ Selesai")
        if hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(100, "✅ Selesai")
            
        # Tampilkan pesan complete dengan delay opsional
        progress_tracker.complete(message, delay)
    else:
        ui_components.get('complete_operation', lambda x: None)(message)

def error_operation_with_message(ui_components: Dict[str, Any], message: str, delay: float = 1.0):
    """Error operation dengan error message dan opsional delay"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Tampilkan pesan error dengan delay opsional
        progress_tracker.error(f"❌ {message}", delay)
    else:
        ui_components.get('error_operation', lambda x: None)(message)

def update_package_status_by_name(ui_components: Dict[str, Any], package_name: str, status: str):
    """Update package status berdasarkan nama dengan category lookup - one-liner"""
    [update_package_status(ui_components, package['key'], status)
     for category in __import__('smartcash.ui.setup.dependency.components.package_selector', fromlist=['get_package_categories']).get_package_categories()
     for package in category['packages']
     if package['pip_name'].split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip().lower() == package_name.lower()]

def batch_update_package_status(ui_components: Dict[str, Any], status_mapping: Dict[str, str]):
    """Batch update package status dengan mapping - one-liner"""
    [update_package_status_by_name(ui_components, package_name, status) 
     for package_name, status in status_mapping.items()]

def get_button_manager_safe(ui_components: Dict[str, Any]):
    """Get button manager dengan safe fallback - one-liner"""
    return ui_components.setdefault('button_manager', 
                                  __import__('smartcash.ui.utils.button_state_manager', fromlist=['get_button_state_manager']).get_button_state_manager(ui_components))

def with_button_context(ui_components: Dict[str, Any], operation: str, func: Callable):
    """Execute function dalam button context dengan automatic state management"""
    button_manager = get_button_manager_safe(ui_components)
    
    try:
        with button_manager.operation_context(operation):
            return func()
    except Exception as e:
        error_operation_with_message(ui_components, f"{operation} failed: {str(e)}")
        raise

def create_progress_tracker(ui_components: Dict[str, Any]) -> Callable:
    """Create progress tracking function dengan closure - one-liner factory"""
    # Fungsi wrapper untuk kompatibilitas dengan kode lama yang menggunakan progress_tracker sebagai callable
    def progress_tracker_wrapper(progress_type, value, message="", color=None):
        # Gunakan metode yang benar sesuai API progress tracker
        update_progress_step(ui_components, progress_type, value, message, color)
    
    return progress_tracker_wrapper

def create_logger_bridge(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Create logger bridge menggunakan built-in logger dari CommonInitializer - one-liner mapping"""
    logger = ui_components.get('logger')
    return {level: lambda msg, lvl=level, log=logger: log and getattr(log, lvl, log.info)(msg) 
            for level in ['info', 'success', 'warning', 'error', 'debug']}

def setup_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Setup progress callback untuk handlers dengan unified interface - one-liner"""
    # Fungsi wrapper untuk kompatibilitas dengan kode lama yang menggunakan progress callback
    def progress_callback_wrapper(**kwargs):
        progress_type = kwargs.get('type', 'overall')
        progress = kwargs.get('progress', 0)
        message = kwargs.get('message', 'Processing...')
        color = kwargs.get('color', None)
        
        # Gunakan metode yang benar sesuai API progress tracker
        update_progress_step(ui_components, progress_type, progress, message, color)
    
    return progress_callback_wrapper

def handle_operation_lifecycle(ui_components: Dict[str, Any], operation_name: str, 
                             operation_func: Callable, *args, **kwargs):
    """Handle complete operation lifecycle dengan error management"""
    
    logger = ui_components.get('logger')
    
    try:
        # Clear outputs dan setup - one-liner
        clear_ui_outputs(ui_components), reset_ui_logger(ui_components), show_operation_progress(ui_components, operation_name)
        
        # Execute operation dalam button context
        result = with_button_context(ui_components, operation_name, lambda: operation_func(*args, **kwargs))
        
        # Success handling - one-liner
        success_msg = f"{operation_name.title()} completed successfully"
        complete_operation_with_message(ui_components, success_msg)
        update_status_panel(ui_components, f"✅ {success_msg}", "success")
        
        return result
        
    except Exception as e:
        # Error handling - one-liner
        error_msg = f"{operation_name.title()} failed: {str(e)}"
        error_operation_with_message(ui_components, error_msg)
        update_status_panel(ui_components, f"❌ {error_msg}", "error")
        if logger:
            logger.error(f"💥 {error_msg}")
        raise

class ProgressSteps:
    """Progress step constants untuk consistent tracking"""
    
    # Installation steps
    INSTALL_INIT = 5
    INSTALL_ANALYSIS = 15
    INSTALL_START = 20
    INSTALL_END = 90
    INSTALL_FINALIZE = 100
    
    # Analysis steps
    ANALYSIS_INIT = 10
    ANALYSIS_GET_PACKAGES = 30
    ANALYSIS_CATEGORIES = 50
    ANALYSIS_CHECK = 60
    ANALYSIS_UPDATE_UI = 90
    ANALYSIS_COMPLETE = 100
    
    # Status check steps
    STATUS_INIT = 10
    STATUS_SYSTEM_INFO = 30
    STATUS_PACKAGE_CHECK = 60
    STATUS_REPORT = 80
    STATUS_UI_UPDATE = 90
    STATUS_COMPLETE = 100

def create_stepped_progress_tracker(ui_components: Dict[str, Any], steps_class=ProgressSteps):
    """Create stepped progress tracker dengan predefined steps - one-liner"""
    
    def step_progress(step_name: str, message: str = "", progress_type: str = "overall"):
        """Update progress menggunakan predefined steps - one-liner"""
        step_value = getattr(steps_class, step_name.upper(), 0)
        
        # Gunakan metode yang benar sesuai API progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if progress_type == 'overall' or progress_type == 'level1':
                progress_tracker.update_overall(step_value, message)
            elif progress_type == 'step' or progress_type == 'level2':
                progress_tracker.update_current(step_value, message)
            
            # Jika update overall, juga update current untuk konsistensi
            if progress_type == 'overall':
                progress_tracker.update_current(step_value, message)
        else:
            # Fallback ke metode lama
            update_progress_step(ui_components, progress_type, step_value, message)
            if progress_type == "overall":
                update_progress_step(ui_components, "step", step_value, message)
    
    return step_progress

def create_operation_context(ui_components: Dict[str, Any], operation_name: str):
    """Create operation context dengan automatic setup dan cleanup"""
    
    class OperationContext:
        def __init__(self, ui_components, operation_name):
            self.ui_components = ui_components
            self.operation_name = operation_name
            self.logger_bridge = create_logger_bridge(ui_components)
            
            # Gunakan progress tracker langsung dari ui_components jika tersedia
            # atau gunakan fungsi wrapper untuk kompatibilitas
            self.progress_tracker = ui_components.get('progress_tracker') or create_progress_tracker(ui_components)
            
            # Tetap sediakan stepped_progress untuk kompatibilitas dengan kode lama
            self.stepped_progress = create_stepped_progress_tracker(ui_components)
        
        def __enter__(self):
            clear_ui_outputs(self.ui_components)
            reset_ui_logger(self.ui_components)
            show_operation_progress(self.ui_components, self.operation_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                error_msg = f"{self.operation_name} failed: {str(exc_val)}"
                error_operation_with_message(self.ui_components, error_msg)
                update_status_panel(self.ui_components, f"❌ {error_msg}", "error")
                
                # Use built-in logger untuk error logging - one-liner
                (logger := self.ui_components.get('logger')) and logger.error(f"💥 {error_msg}")
            return False  # Don't suppress exceptions
    
    return OperationContext(ui_components, operation_name)

# Consolidated utility functions - one-liner style
def log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Safe logging menggunakan built-in logger dari CommonInitializer - one-liner"""
    (logger := ui_components.get('logger')) and getattr(logger, level, logger.info)(message)

def safe_execute_with_logging(ui_components: Dict[str, Any], func: Callable, error_msg: str = "Operation failed"):
    """Safe execution dengan error logging - one-liner"""
    try:
        return func() if callable(func) else None
    except Exception as e:
        log_to_ui_safe(ui_components, f"{error_msg}: {str(e)}", "error")
        return None

# One-liner factory functions
create_safe_progress_updater = lambda ui_components: lambda ptype, val, msg="", color=None: update_progress_step(ui_components, ptype, val, msg, color)
create_safe_status_updater = lambda ui_components: lambda msg, stype="info": update_status_panel(ui_components, msg, stype)
create_safe_logger = lambda ui_components: lambda msg, level="info": log_to_ui_safe(ui_components, msg, level)