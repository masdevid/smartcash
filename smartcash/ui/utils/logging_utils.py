"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas untuk mengatur logging di notebook dengan integrasi UI
"""

import logging
import sys
import threading
from typing import Dict, Any, Optional
from IPython.display import display, HTML

def setup_ipython_logging(ui_components: Dict[str, Any], 
                         module_name: Optional[str] = None, 
                         log_to_file: bool = False,
                         log_dir: str = "logs",
                         log_level: int = logging.INFO,
                         redirect_all_logs: bool = False) -> Any:
    """
    Setup logger untuk IPython notebook dengan output ke UI widget.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        module_name: Nama modul untuk logger (opsional)
        log_to_file: Flag untuk mengaktifkan logging ke file
        log_dir: Direktori untuk menyimpan file log
        log_level: Level logging (default: INFO)
        redirect_all_logs: Flag untuk mengalihkan semua console log ke output log UI
        
    Returns:
        Logger yang dikonfigurasi atau None jika gagal
    """
    try:
        # Gunakan nama modul dari ui_components jika tersedia dan tidak ada parameter
        if not module_name and 'module_name' in ui_components:
            module_name = ui_components['module_name']
        
        # Default ke 'ipython' jika masih tidak ada nama modul
        module_name = module_name or 'ipython'
        
        # Import UILogger
        from smartcash.ui.utils.ui_logger import create_ui_logger
        
        # Buat UI logger yang terintegrasi
        logger = create_ui_logger(
            ui_components, 
            name=module_name, 
            log_to_file=log_to_file,
            log_dir=log_dir,
            log_level=log_level
        )
        
        # Setup observer integration
        try:
            _setup_observer_integration_minimal(ui_components)
        except Exception:
            pass
        
        # Register cleanup function
        register_cleanup_on_cell_execution(ui_components)
        
        # Redirect semua log ke UI jika diminta
        if redirect_all_logs:
            redirect_all_logs_to_ui(ui_components)
        
        return logger
        
    except Exception as e:
        # Fallback minimal: tampilkan error di UI dan kembalikan logger standar
        error_message = f"Error setup logger: {str(e)}"
        
        if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
            with ui_components['status']:
                display(HTML(f"<div style='color:orange'>{error_message}</div>"))
        else:
            print(f"{error_message}")
        
        # Return standard logger
        return logging.getLogger(module_name or 'ipython')

def redirect_all_logs_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Alihkan semua console log ke output log UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    if 'logger' not in ui_components:
        print("⚠️ Tidak dapat mengalihkan log: logger tidak ditemukan di ui_components")
        return
    
    # Dapatkan UI logger
    ui_logger = ui_components['logger']
    
    # Redirect stdout/stderr ke UI
    from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
    intercept_stdout_to_ui(ui_components)
    
    # Redirect semua logger ke UI logger
    redirect_all_loggers_to_ui_logger(ui_logger)
    
    # Simpan flag bahwa semua log telah dialihkan
    ui_components['all_logs_redirected'] = True
    
    print("✅ Semua console log berhasil dialihkan ke output log UI")

def redirect_all_loggers_to_ui_logger(ui_logger) -> None:
    """
    Alihkan semua logger ke UI logger.
    
    Args:
        ui_logger: Instance UILogger
    """
    # Dapatkan root logger
    root_logger = logging.getLogger()
    
    # Hapus semua handler yang ada
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Buat custom handler yang mengalihkan ke UI logger
    class UILogHandler(logging.Handler):
        def __init__(self, ui_logger):
            super().__init__()
            self.ui_logger = ui_logger
        
        def emit(self, record):
            try:
                # Format pesan
                msg = self.format(record)
                
                # Log ke UI logger berdasarkan level
                if record.levelno >= logging.CRITICAL:
                    self.ui_logger.critical(msg)
                elif record.levelno >= logging.ERROR:
                    self.ui_logger.error(msg)
                elif record.levelno >= logging.WARNING:
                    self.ui_logger.warning(msg)
                else:
                    self.ui_logger.info(msg)
            except Exception:
                self.handleError(record)
    
    # Buat dan tambahkan handler ke root logger
    handler = UILogHandler(ui_logger)
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set level ke DEBUG agar semua log ditangkap
    root_logger.setLevel(logging.DEBUG)
    
    # Alihkan juga SmartCashLogger jika ada
    try:
        from smartcash.common.logger import get_logger, LogLevel
        
        # Dapatkan instance SmartCashLogger
        sc_logger = get_logger()
        
        # Tambahkan callback untuk mengalihkan log ke UI logger
        def ui_log_callback(level, message):
            if level.name == 'DEBUG':
                ui_logger.debug(message)
            elif level.name == 'INFO':
                ui_logger.info(message)
            elif level.name == 'SUCCESS':
                ui_logger.success(message)
            elif level.name == 'WARNING':
                ui_logger.warning(message)
            elif level.name in ('ERROR', 'CRITICAL'):
                ui_logger.error(message)
        
        # Tambahkan callback ke SmartCashLogger
        sc_logger.add_callback(ui_log_callback)
    except ImportError:
        # Tidak ada SmartCashLogger, lewati
        pass

def restore_console_logs(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan console log ke aslinya.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Kembalikan stdout/stderr ke aslinya
    from smartcash.ui.utils.ui_logger import restore_stdout
    restore_stdout(ui_components)
    
    # Reset semua logging handler
    from smartcash.ui.utils.ui_logger import _reset_logging_handlers
    _reset_logging_handlers()
    
    # Hapus flag
    if 'all_logs_redirected' in ui_components:
        del ui_components['all_logs_redirected']
    
    print("✅ Console log dikembalikan ke aslinya")

def _setup_observer_integration_minimal(ui_components: Dict[str, Any]) -> None:
    """
    Siapkan integrasi dengan observer pattern minimal tanpa log yang berlebihan.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Import observer handler tanpa circular imports
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        
        # Set group name dari module_name atau default
        observer_group = f"{ui_components.get('module_name', 'default')}_observers"
        
        # Setup observers dengan group yang spesifik untuk cell ini
        ui_components_updated = setup_observer_handlers(ui_components, observer_group)
        
        # Update ui_components dengan observer manager dan group
        for key in ['observer_manager', 'observer_group']:
            if key in ui_components_updated and key not in ui_components:
                ui_components[key] = ui_components_updated[key]
    except Exception:
        # Tidak perlu log error - observer adalah opsional
        pass

def create_cleanup_function(ui_components: Dict[str, Any]) -> callable:
    """
    Buat fungsi cleanup untuk resource logger dan observer.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Callable untuk membersihkan resources
    """
    def cleanup():
        """Membersihkan resources saat sel baru dieksekusi."""
        # Reset stdout jika diintercepti
        try:
            from smartcash.ui.utils.ui_logger import restore_stdout
            restore_stdout(ui_components)
        except ImportError:
            # Fallback jika tidak bisa import restore_stdout
            if 'original_stdout' in ui_components:
                sys.stdout = ui_components['original_stdout']
                ui_components.pop('original_stdout', None)
        
        # Reset semua logging handler jika semua log telah dialihkan
        if 'all_logs_redirected' in ui_components:
            try:
                from smartcash.ui.utils.ui_logger import _reset_logging_handlers
                _reset_logging_handlers()
                ui_components.pop('all_logs_redirected', None)
            except ImportError:
                pass
        
        # Unregister observer jika ada
        if 'observer_manager' in ui_components and 'observer_group' in ui_components:
            try:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            except Exception:
                pass
        
        # Clear resources yang terdaftar
        if 'resources' in ui_components:
            for resource, cleanup_func in ui_components['resources']:
                try:
                    if cleanup_func and callable(cleanup_func):
                        cleanup_func(resource)
                    elif hasattr(resource, 'close') and callable(resource.close):
                        resource.close()
                except Exception:
                    pass
            ui_components['resources'] = []
        
        # Reset progress tracker jika ada
        for tracker_key in [k for k in ui_components if k.endswith('_tracker')]:
            try:
                tracker = ui_components[tracker_key]
                if hasattr(tracker, 'complete'):
                    tracker.complete("Dibersihkan")
            except Exception:
                pass
        
        # Reset progress UI
        for key in ['progress_bar', 'progress_message']:
            if key in ui_components and hasattr(ui_components[key], 'layout'):
                ui_components[key].layout.visibility = 'hidden'
    
    return cleanup

def register_cleanup_on_cell_execution(ui_components: Dict[str, Any]) -> None:
    """
    Register cleanup function ke IPython untuk dijalankan sebelum menjalankan cell.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        # Buat cleanup function
        cleanup_func = create_cleanup_function(ui_components)
        ui_components['cleanup'] = cleanup_func
        
        # Unregister existing handlers terlebih dahulu untuk mencegah duplikasi
        if ipython and hasattr(ipython.events, '_events'):
            for event_type in ipython.events._events:
                if event_type == 'pre_run_cell':
                    existing_handlers = ipython.events._events[event_type]
                    for handler in list(existing_handlers):
                        if handler.__qualname__.endswith('cleanup'):
                            try:
                                ipython.events.unregister('pre_run_cell', handler)
                            except:
                                pass
        
        # Register ke event IPython
        if ipython:
            ipython.events.register('pre_run_cell', cleanup_func)
    except (ImportError, AttributeError):
        # Tidak dalam IPython environment
        pass