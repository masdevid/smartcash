"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas untuk setup logging dengan arahkan output ke UI komponen
"""

import logging
import sys
from typing import Dict, Any, Optional
from IPython.display import display, HTML

def setup_ipython_logging(ui_components: Dict[str, Any], module_name: Optional[str] = None) -> Optional[logging.Logger]:
    """
    Setup logger untuk IPython notebook dengan output ke UI widget.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        module_name: Nama modul untuk logger (opsional)
        
    Returns:
        Logger yang dikonfigurasi atau None jika gagal
    """
    try:
        from smartcash.ui.utils.ui_logger import redirect_logger_to_ui, log_to_ui
        from smartcash.common.logger import get_logger
        
        # Gunakan nama modul dari ui_components jika tersedia dan tidak ada parameter
        if not module_name and 'module_name' in ui_components:
            module_name = ui_components['module_name']
        
        # Default ke 'ipython' jika masih tidak ada nama modul
        module_name = module_name or 'ipython'
        
        # Dapatkan logger
        logger = get_logger(module_name)
        
        # Arahkan ke UI
        if redirect_logger_to_ui(logger, ui_components):
            log_to_ui(ui_components, f"Logger {module_name} berhasil terarah ke UI output", "info", "🔄")
        
        return logger
        
    except Exception as e:
        # Fallback ke standard logger jika gagal
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<div style='color:orange'>⚠️ Error setup logger: {str(e)}</div>"))
        return None

def setup_ui_console_logger(ui_components: Dict[str, Any], module_name: Optional[str] = None) -> Optional[logging.Logger]:
    """
    Setup logger dengan output ke UI dan console.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        module_name: Nama modul untuk logger
        
    Returns:
        Logger yang dikonfigurasi atau None jika gagal
    """
    try:
        logger = setup_ipython_logging(ui_components, module_name)
        
        if logger:
            # Tambahkan handler console
            console_handler = logging.StreamHandler(sys.stdout)
            console_format = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)
            
            return logger
        
        return None
    
    except Exception:
        return None

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """
    Log pesan langsung ke UI tanpa melalui logger.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, debug, success)
        emoji: Emoji untuk ditampilkan (opsional)
    """
    try:
        # Try to import dedicated ui_logger
        from smartcash.ui.utils.ui_logger import log_to_ui as ui_logger_log_to_ui
        ui_logger_log_to_ui(ui_components, message, level, emoji)
        return
    except ImportError:
        # Fallback implementation
        if 'status' not in ui_components:
            return
            
        # Map level ke warna
        color_map = {
            'debug': 'gray',
            'info': 'blue',
            'success': 'green',
            'warning': 'orange',
            'error': 'red'
        }
        
        # Default emoji berdasarkan level jika tidak disediakan
        if not emoji:
            emoji_map = {
                'debug': '🐞',
                'info': 'ℹ️',
                'success': '✅',
                'warning': '⚠️',
                'error': '❌'
            }
            emoji = emoji_map.get(level, '')
        
        # Get color
        color = color_map.get(level, 'black')
        
        # Create HTML
        html_msg = f"""<div style="margin: 2px 0; color: {color}; overflow-wrap: break-word;">
            <span>{emoji} <b>{level.upper()}:</b></span> {message}
        </div>"""
        
        # Display in widget
        with ui_components['status']:
            display(HTML(html_msg))

def add_file_handler(logger: logging.Logger, file_path: str, level: int = logging.DEBUG) -> None:
    """
    Tambahkan file handler ke logger.
    
    Args:
        logger: Logger yang akan ditambahkan handler
        file_path: Path file log
        level: Level logging (default: DEBUG)
    """
    try:
        # Buat handler
        handler = logging.FileHandler(file_path)
        handler.setLevel(level)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Tambahkan handler ke logger
        logger.addHandler(handler)
    except Exception:
        pass

def reset_logging():
    """Reset semua konfigurasi logging."""
    logging.shutdown()
    root_logger = logging.getLogger()
    
    # Hapus semua handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)