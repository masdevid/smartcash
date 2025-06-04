"""
File: smartcash/ui/setup/dependency/utils/ui_state_utils.py
Deskripsi: Cleaned UI state management utilities tanpa log_message_safe dan duplicate logging
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.setup.dependency.components.package_selector import update_package_status

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
    (log_output := ui_components.get('log_output')) and hasattr(log_output, 'clear_output') and log_output.clear_output(wait=True)

def update_progress_step(ui_components: Dict[str, Any], progress_type: str, value: int, 
                        message: str = "", color: str = None):
    """Update progress dengan consolidated approach - one-liner"""
    ui_components.get('update_progress', lambda *a: None)(progress_type, value, message, color)

def show_operation_progress(ui_components: Dict[str, Any], operation: str):
    """Show progress container untuk operation - one-liner"""
    ui_components.get('show_for_operation', lambda x: None)(operation)

def complete_operation_with_message(ui_components: Dict[str, Any], message: str):
    """Complete operation dengan success message - one-liner"""
    ui_components.get('complete_operation', lambda x: None)(message)

def error_operation_with_message(ui_components: Dict[str, Any], message: str):
    """Error operation dengan error message - one-liner"""
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
    return lambda progress_type, value, message="", color=None: update_progress_step(ui_components, progress_type, value, message, color)

def create_logger_bridge(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Create logger bridge menggunakan built-in logger dari CommonInitializer - one-liner mapping"""
    logger = ui_components.get('logger')
    return {level: lambda msg, lvl=level, log=logger: log and getattr(log, lvl, log.info)(msg) 
            for level in ['info', 'success', 'warning', 'error', 'debug']}

def setup_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Setup progress callback untuk handlers dengan unified interface - one-liner"""
    return lambda **kwargs: update_progress_step(
        ui_components, kwargs.get('type', 'overall'), kwargs.get('progress', 0), 
        kwargs.get('message', 'Processing...'), kwargs.get('color', None)
    )

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
        logger and logger.error(f"💥 {error_msg}")
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
        update_progress_step(ui_components, progress_type, step_value, message)
        progress_type == "overall" and update_progress_step(ui_components, "step", step_value, message)
    
    return step_progress

def create_operation_context(ui_components: Dict[str, Any], operation_name: str):
    """Create operation context dengan automatic setup dan cleanup"""
    
    class OperationContext:
        def __init__(self, ui_components, operation_name):
            self.ui_components = ui_components
            self.operation_name = operation_name
            self.logger_bridge = create_logger_bridge(ui_components)
            self.progress_tracker = create_progress_tracker(ui_components)
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