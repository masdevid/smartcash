"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas untuk mengarahkan logging ke UI dengan kompatibilitas SmartCashLogger
"""

import logging
import sys
from typing import Dict, Any, Optional
from IPython.display import display, HTML

# Import komponen dari ui_logger
try:
    from smartcash.ui.utils.ui_logger import create_direct_ui_logger, log_to_ui, intercept_cell_utils_logs
except ImportError:
    # Definisikan fungsi placeholder jika import gagal
    def log_to_ui(ui_components, message, level="info", emoji=""):
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<div>{message}</div>"))
    
    def create_direct_ui_logger(ui_components, name):
        return logging.getLogger(name)
    
    def intercept_cell_utils_logs(ui_components):
        pass

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
        
        # Buat logger yang langsung ke UI
        logger = create_direct_ui_logger(ui_components, module_name)
        
        # Log sukses
        logger.info(f"Logger {module_name} terinisialisasi")
        
        return logger
        
    except Exception as e:
        # Fallback ke standard logger jika gagal
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<div style='color:orange'>⚠️ Error setup logger: {str(e)}</div>"))
        
        # Coba fallback ke logger standard
        try:
            return logging.getLogger(module_name or 'ipython')
        except:
            return None

def capture_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Tangkap output stdout dan arahkan ke UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Gunakan fungsi dari ui_logger
    intercept_cell_utils_logs(ui_components)

def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan stdout original.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    if 'original_stdout' in ui_components:
        sys.stdout = ui_components['original_stdout']

def reset_logging():
    """Reset semua konfigurasi logging."""
    logging.shutdown()
    root_logger = logging.getLogger()
    
    # Hapus semua handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)