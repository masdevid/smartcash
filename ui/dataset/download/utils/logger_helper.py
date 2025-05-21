"""
File: smartcash/ui/dataset/download/utils/logger_helper.py
Deskripsi: Helper untuk penggunaan UI logger yang konsisten di module download
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

# Constant untuk nama modul logger
MODULE_LOGGER_NAME = 'smartcash.dataset.download'

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke UI dan logger Python dengan konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional untuk ditampilkan di depan pesan
    """
    # Cek apakah ini adalah dependency installer
    # Jika ui_components tidak memiliki flag download_initialized, 
    # maka kemungkinan ini adalah dependency installer atau saat impor
    if not is_initialized(ui_components):
        # Skip UI logging jika belum diinisialisasi untuk mencegah log muncul di dependency installer
        return
    
    logger = ui_components.get('logger')
    
    # Log ke UI hanya jika log_output atau output tersedia
    if 'log_output' in ui_components or 'output' in ui_components:
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
        # Gunakan nama modul yang spesifik untuk menghindari conflict
        ui_components['logger'] = get_logger(MODULE_LOGGER_NAME)
    
    # Tambahkan fungsi log_message ke UI components
    ui_components['log_message'] = lambda msg, level="info", icon=None: log_message(ui_components, msg, level, icon)
    
    # Tambahkan flag download_initialized untuk menunjukkan modul telah di-initialize
    # Flag ini digunakan untuk mencegah log ke installer dependency
    ui_components['download_initialized'] = True
    
    return ui_components 

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI logger sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika sudah diinisialisasi, False jika belum
    """
    return ui_components.get('download_initialized', False) 