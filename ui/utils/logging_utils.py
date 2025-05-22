"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas logging yang disederhanakan dan aligned dengan ui_logger.py terbaru
"""

import logging
import sys
from typing import Dict, Any, Optional
from IPython.display import display, HTML

def setup_ipython_logging(ui_components: Dict[str, Any], 
                         module_name: Optional[str] = None, 
                         log_to_file: bool = False,
                         log_dir: str = "logs",
                         log_level: int = logging.INFO,
                         redirect_all_logs: bool = False) -> Any:
    """
    Setup logger untuk IPython dengan integrasi UI yang disederhanakan
    
    Args:
        ui_components: Dictionary komponen UI
        module_name: Nama modul untuk logger
        log_to_file: Flag untuk logging ke file
        log_dir: Direktori log files
        log_level: Level logging
        redirect_all_logs: Redirect semua logs ke UI
        
    Returns:
        UILogger instance atau None jika gagal
    """
    try:
        # Import UILogger
        from smartcash.ui.utils.ui_logger import create_ui_logger
        
        # Set module name
        if not module_name and 'module_name' in ui_components:
            module_name = ui_components['module_name']
        module_name = module_name or 'ipython'
        
        # Create UI logger
        logger = create_ui_logger(
            ui_components, 
            name=module_name, 
            log_to_file=log_to_file,
            redirect_stdout=redirect_all_logs,
            log_dir=log_dir,
            log_level=log_level
        )
        
        # Register cleanup untuk cell execution
        register_cleanup_on_cell_execution(ui_components)
        
        return logger
        
    except Exception as e:
        # Fallback dengan error display
        error_message = f"Error setup logger: {str(e)}"
        
        if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
            with ui_components['status']:
                display(HTML(f"<div style='color:orange'>{error_message}</div>"))
        
        return logging.getLogger(module_name or 'ipython')

def redirect_all_logs_to_ui(ui_components: Dict[str, Any]) -> None:
    """Redirect semua console logs ke UI logger"""
    if 'logger' not in ui_components:
        return
    
    try:
        from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
        intercept_stdout_to_ui(ui_components)
        ui_components['all_logs_redirected'] = True
    except ImportError:
        pass

def restore_console_logs(ui_components: Dict[str, Any]) -> None:
    """Restore console logs ke original state"""
    try:
        from smartcash.ui.utils.ui_logger import restore_stdout
        restore_stdout(ui_components)
        
        if 'all_logs_redirected' in ui_components:
            del ui_components['all_logs_redirected']
    except ImportError:
        pass

def create_cleanup_function(ui_components: Dict[str, Any]) -> callable:
    """Create cleanup function untuk resources"""
    def cleanup():
        """Clean up resources saat cell baru dieksekusi"""
        # Restore stdout jika di-intercept
        try:
            restore_console_logs(ui_components)
        except Exception:
            if 'original_stdout' in ui_components:
                sys.stdout = ui_components['original_stdout']
                ui_components.pop('original_stdout', None)
        
        # Clean observer jika ada
        if 'observer_manager' in ui_components and 'observer_group' in ui_components:
            try:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            except Exception:
                pass
        
        # Clean resources
        if 'resources' in ui_components:
            for resource, cleanup_func in ui_components['resources']:
                try:
                    if cleanup_func and callable(cleanup_func):
                        cleanup_func(resource)
                    elif hasattr(resource, 'close'):
                        resource.close()
                except Exception:
                    pass
            ui_components['resources'] = []
        
        # Hide progress UI
        for key in ['progress_bar', 'progress_message', 'progress_container']:
            if key in ui_components and hasattr(ui_components[key], 'layout'):
                ui_components[key].layout.visibility = 'hidden'
    
    return cleanup

def register_cleanup_on_cell_execution(ui_components: Dict[str, Any]) -> None:
    """Register cleanup function ke IPython events"""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        if not ipython:
            return
        
        cleanup_func = create_cleanup_function(ui_components)
        ui_components['cleanup'] = cleanup_func
        
        # Unregister existing cleanup handlers
        if hasattr(ipython.events, '_events') and 'pre_run_cell' in ipython.events._events:
            existing_handlers = list(ipython.events._events['pre_run_cell'])
            for handler in existing_handlers:
                if hasattr(handler, '__qualname__') and 'cleanup' in handler.__qualname__:
                    try:
                        ipython.events.unregister('pre_run_cell', handler)
                    except:
                        pass
        
        # Register new cleanup
        ipython.events.register('pre_run_cell', cleanup_func)
        
    except (ImportError, AttributeError):
        pass