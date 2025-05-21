"""
File: smartcash/ui/dataset/download/utils/logger_helper.py
Deskripsi: Helper untuk penggunaan UI logger yang konsisten di module download
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke UI dan logger Python dengan konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional untuk ditampilkan di depan pesan
    """
    logger = ui_components.get('logger')
    
    # Log ke UI dengan konsisten menggunakan UI logger
    ui_log(ui_components, message, level, icon)
    
    # Log ke Python logger jika tersedia
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "warning" or level == "warn":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        elif level == "success":
            # Success level tidak ada di Python logger standard, gunakan info
            logger.info(f"✅ {message}")
        elif level == "critical":
            logger.critical(message)

def setup_ui_logger(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup UI logger untuk module download.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah ditambahkan logger
    """
    # Setup logger jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = get_logger()
    
    # Tambahkan fungsi log_message ke UI components
    ui_components['log_message'] = lambda msg, level="info", icon=None: log_message(ui_components, msg, level, icon)
    
    return ui_components 