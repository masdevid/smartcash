"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Fixed logging utilities tanpa tqdm suppression untuk avoid AttributeError
"""

import logging
import sys
import warnings
import os
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
from IPython.display import display, HTML

# Type aliases for better type hints
T = TypeVar('T')
LoggerType = logging.Logger


def setup_aggressive_log_suppression() -> None:
    """Setup aggressive log suppression to prevent pollution from backend services"""
    # Clear and disable root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.CRITICAL)
    root.propagate = False
    
    # Add NullHandler to prevent "No handlers could be found" warnings
    if not root.handlers:
        root.addHandler(logging.NullHandler())
    
    # Extended list of loggers to suppress
    suppression_targets = [
        # Core Python
        '', 'root', '__main__',
        
        # Common libraries
        'requests', 'urllib3', 'urllib', 'http', 'httpx',
        'asyncio', 'concurrent', 'threading', 'multiprocessing',
        'tornado', 'traitlets', 'zmq', 'jupyter', 'IPython',
        
        # Data science
        'numpy', 'pandas', 'scipy', 'h5py', 'tables', 'pytables',
        'matplotlib', 'PIL', 'Pillow', 'seaborn', 'plotly', 'bokeh',
        'altair', 'plotnine', 'ggplot', 'sklearn', 'tensorflow',
        'torch', 'transformers', 'datasets', 'tqdm', 'tqdm.auto',
        
        # Web and UI
        'ipywidgets', 'ipykernel', 'ipython', 'jupyter_client',
        'jupyter_core', 'notebook', 'nbformat', 'nbconvert',
        
        # SmartCash modules
        'smartcash', 'smartcash.*',  # This will match all submodules
    ]
    
    # Suppress all targets with safe error handling
    for target in suppression_targets:
        try:
            logger = logging.getLogger(target)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
            logger.handlers = []
            # Add NullHandler to prevent "No handlers could be found" warnings
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
        except Exception:
            pass  # Silent fail for compatibility
    
    # Suppress warnings globally
    warnings.filterwarnings('ignore')
    # Also suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Suppress specific warnings from common libraries
    for warning_category in [
        'ignore:The psycopg2 wheel package will be renamed',
        'ignore:distutils Version classes are deprecated',
        'ignore:There is no current event loop',
        'ignore:Jupyter is migrating its paths',
        'ignore:Setuptools is replacing distutils',
        'ignore:numpy.ufunc size changed',
        'ignore:invalid value encountered',
        'ignore:divide by zero encountered',
    ]:
        warnings.filterwarnings("ignore", message=warning_category)

def setup_stdout_suppression() -> None:
    """Setup stdout/stderr suppression dengan anonymous class pattern"""
    if not hasattr(sys, '_original_stdout_saved'):
        sys._original_stdout_saved = sys.stdout
        sys.stdout = type('StdoutSuppressor', (), {
            'write': lambda self, x: None,
            'flush': lambda self: None,
            'isatty': lambda self: False,
            'fileno': lambda self: sys._original_stdout_saved.fileno()
        })()
    
    if not hasattr(sys, '_original_stderr_saved'):
        sys._original_stderr_saved = sys.stderr
        sys.stderr = type('StderrSuppressor', (), {
            'write': lambda self, x: None,
            'flush': lambda self: None,
            'isatty': lambda self: False,
            'fileno': lambda self: sys._original_stderr_saved.fileno()
        })()

def restore_stdout() -> None:
    """Restore stdout/stderr ke original state"""
    if hasattr(sys, '_original_stdout_saved'):
        sys.stdout = sys._original_stdout_saved
        delattr(sys, '_original_stdout_saved')
    if hasattr(sys, '_original_stderr_saved'):
        sys.stderr = sys._original_stderr_saved
        delattr(sys, '_original_stderr_saved')

def setup_ipython_logging(ui_components: Dict[str, Any], 
                         module_name: Optional[str] = None, 
                         log_to_file: bool = False,
                         log_dir: str = "logs",
                         log_level: int = logging.INFO,
                         redirect_all_logs: bool = False) -> Any:
    """Setup logger untuk IPython dengan integrasi UI yang disederhanakan tanpa tqdm manipulation"""
    try:
        setup_aggressive_log_suppression()
        if redirect_all_logs:
            setup_stdout_suppression()
        
        from smartcash.ui.utils.ui_logger import create_ui_logger
        
        module_name = module_name or ui_components.get('module_name', 'ipython')
        
        logger = create_ui_logger(
            ui_components, 
            name=module_name, 
            log_to_file=log_to_file,
            redirect_stdout=redirect_all_logs,
            log_dir=log_dir,
            log_level=log_level
        )
        
        register_cleanup_on_cell_execution(ui_components)
        return logger
        
    except Exception as e:
        error_message = f"🚨 Error setup logger: {str(e)}"
        
        if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
            with ui_components['status']:
                display(HTML(f"<div style='color:orange'>{error_message}</div>"))
        
        return logging.getLogger(module_name or 'ipython')

def register_cleanup_on_cell_execution(ui_components: Dict[str, Any]) -> None:
    """Register cleanup function ke IPython events tanpa tqdm manipulation"""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        if not ipython:
            return
        
        def cleanup_func():
            restore_stdout()
            
            # Clean resources safely
            resources = ui_components.get('resources', [])
            for resource, cleanup_func in resources:
                try:
                    if callable(cleanup_func):
                        cleanup_func(resource)
                    elif hasattr(resource, 'close'):
                        resource.close()
                except Exception:
                    pass
            
            ui_components['resources'] = []
            
            # Hide UI elements safely
            for key in ['progress_bar', 'progress_message', 'progress_container']:
                widget = ui_components.get(key)
                if widget and hasattr(widget, 'layout'):
                    try:
                        widget.layout.visibility = 'hidden'
                    except Exception:
                        pass
        
        # Unregister existing handlers safely
        try:
            if hasattr(ipython.events, '_events') and 'pre_run_cell' in ipython.events._events:
                existing_handlers = list(ipython.events._events['pre_run_cell'])
                for handler in existing_handlers:
                    if hasattr(handler, '__qualname__') and 'cleanup' in handler.__qualname__:
                        ipython.events.unregister('pre_run_cell', handler)
        except Exception:
            pass
        
        ui_components['cleanup'] = cleanup_func
        ipython.events.register('pre_run_cell', cleanup_func)
        
    except (ImportError, AttributeError):
        pass

def allow_tqdm_display() -> None:
    """Allow tqdm untuk display normal progress bars tanpa manipulation yang bermasalah"""
    try:
        import tqdm
        # Restore tqdm ke behavior normal jika sebelumnya di-suppress
        if hasattr(tqdm.tqdm, '_original_init'):
            tqdm.tqdm.__init__ = tqdm.tqdm._original_init
        if hasattr(tqdm.tqdm, '_original_update'):
            tqdm.tqdm.update = tqdm.tqdm._original_update
        if hasattr(tqdm.tqdm, '_original_close'):
            tqdm.tqdm.close = tqdm.tqdm._original_close
    except ImportError:
        pass

# One-liner utilities dengan safe tqdm handling
redirect_all_logs_to_ui = lambda ui_components: setup_stdout_suppression() if ui_components else None
restore_console_logs = lambda ui_components: restore_stdout() if ui_components else None
suppress_backend_logs = lambda: setup_aggressive_log_suppression()
suppress_all_outputs = lambda: (setup_aggressive_log_suppression(), setup_stdout_suppression())

def get_clean_logger(module_name: str = "clean_ui") -> logging.Logger:
    """Get logger dengan suppression sudah applied tanpa tqdm issues"""
    setup_aggressive_log_suppression()
    return logging.getLogger(module_name)

def create_silent_context():
    """Context manager untuk operasi silent tanpa logs dan tanpa tqdm manipulation"""
    class SilentContext:
        def __enter__(self):
            setup_aggressive_log_suppression()
            setup_stdout_suppression()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            restore_stdout()
    
    return SilentContext()

def log_missing_components(components: Dict[str, Any], exclude: list = None) -> None:
    """Log a warning for any None values in the components dictionary.
    
    Args:
        components: Dictionary of component names and their values
        exclude: List of component names to exclude from the check
    """
    if exclude is None:
        exclude = ['logger']
    
    missing = [k for k, v in components.items() if v is None and k not in exclude]
    if missing:
        logger = logging.getLogger(__name__)
        logger.warning("Missing optional UI components: %s", 
                     ", ".join(f"'{m}'" for m in missing),
                     extra={"missing_components": missing})

# Specific suppression functions dengan safe error handling
def suppress_ml_logs() -> None:
    """Suppress ML library logs dengan safe handling"""
    for lib in ['tensorflow', 'torch', 'sklearn', 'cv2']:
        try:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        except Exception:
            pass

def suppress_viz_logs() -> None:
    """Suppress visualization library logs dengan safe handling"""
    for lib in ['matplotlib', 'seaborn', 'plotly', 'bokeh']:
        try:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        except Exception:
            pass

def suppress_data_logs() -> None:
    """Suppress data processing library logs dengan safe handling"""
    for lib in ['pandas', 'numpy', 'scipy', 'h5py']:
        try:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
        except Exception:
            pass