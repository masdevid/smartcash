"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas terpadu untuk mengarahkan logging ke UI widget dengan integrasi ke observer pattern
"""

import logging
import sys
from typing import Dict, Any, Optional
from IPython.display import display, HTML

def setup_ipython_logging(ui_components: Dict[str, Any], module_name: Optional[str] = None) -> Any:
    """
    Setup logger untuk IPython notebook dengan output ke UI widget.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        module_name: Nama modul untuk logger (opsional)
        
    Returns:
        Logger yang dikonfigurasi atau None jika gagal
    """
    try:
        # Gunakan nama modul dari ui_components jika tersedia dan tidak ada parameter
        if not module_name and 'module_name' in ui_components:
            module_name = ui_components['module_name']
        
        # Default ke 'ipython' jika masih tidak ada nama modul
        module_name = module_name or 'ipython'
        
        # Import komponen UI logger
        from smartcash.ui.utils.ui_logger import create_direct_ui_logger, intercept_stdout_to_ui
        
        # Pastikan intercept stdout dilakukan terlebih dahulu, sebelum membuat logger
        intercept_stdout_to_ui(ui_components)
        
        # Buat logger yang langsung ke UI
        logger = create_direct_ui_logger(ui_components, module_name)
        
        # Log success
        logger.info(f"Logger {module_name} terinisialisasi")
        
        # Setup observer integration jika tersedia
        setup_observer_integration(ui_components, logger)
        
        return logger
        
    except Exception as e:
        # Fallback minimal: tampilkan error di UI dan kembalikan logger standar
        error_message = f"Error setup logger: {str(e)}"
        
        if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
            with ui_components['status']:
                display(HTML(f"<div style='color:orange'>⚠️ {error_message}</div>"))
        else:
            print(f"⚠️ {error_message}")
        
        # Return standard logger
        return logging.getLogger(module_name or 'ipython')


def setup_observer_integration(ui_components: Dict[str, Any], logger: Any = None) -> None:
    """
    Siapkan integrasi dengan observer pattern jika tersedia.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        logger: Logger untuk pesan error/info
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
        
        # Log success jika logger ada
        if logger:
            logger.debug(f"👁️ Observer integration berhasil disetup dengan group '{observer_group}'")
    except (ImportError, AttributeError):
        # Tidak ada observer handler, tidak masalah
        if logger:
            logger.debug("ℹ️ Observer handler tidak tersedia, integrasi observer tidak diaktifkan")
    except Exception as e:
        # Error lain saat setup observer
        if logger:
            logger.warning(f"⚠️ Error saat setup observer integration: {str(e)}")


def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan stdout ke aslinya.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    if 'original_stdout' in ui_components:
        sys.stdout = ui_components['original_stdout']
        ui_components.pop('custom_stdout', None)


def reset_logging() -> None:
    """Reset semua konfigurasi logging."""
    # Reset root logger
    logging.shutdown()
    root_logger = logging.getLogger()
    
    # Hapus semua handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Reset level
    root_logger.setLevel(logging.INFO)


def create_cleanup_function(ui_components: Dict[str, Any]) -> callable:
    """
    Buat fungsi cleanup untuk resource logger dan observer.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Callable untuk membersihkan resources
    """
    def cleanup():
        # Reset stdout jika diintercepti
        restore_stdout(ui_components)
        
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
        
        # Register ke event IPython
        if ipython:
            ipython.events.register('pre_run_cell', cleanup_func)
    except (ImportError, AttributeError):
        # Tidak dalam IPython environment
        pass