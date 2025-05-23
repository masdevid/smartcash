"""
File: smartcash/ui/dataset/download/utils/logger_helper.py
Deskripsi: Helper untuk penggunaan UI logger yang konsisten di module download
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

# Import namespace konstanta
from smartcash.ui.dataset.download.download_initializer import DOWNLOAD_LOGGER_NAMESPACE, MODULE_LOGGER_NAME

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke UI dan logger Python dengan namespace khusus dataset download.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional untuk ditampilkan di depan pesan
    """
    # Cek apakah ini adalah dataset download yang sudah diinisialisasi
    if not is_initialized(ui_components):
        # Skip UI logging jika belum diinisialisasi untuk mencegah log muncul di modul lain
        return
    
    # Pastikan menggunakan logger dengan namespace yang tepat
    logger = ui_components.get('logger') or get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    # Log ke UI hanya jika log_output atau output tersedia
    if 'log_output' in ui_components or 'output' in ui_components:
        # Log ke UI dengan konsisten menggunakan UI logger
        ui_log(ui_components, message, level, icon)
    
    # Tambahkan prefix untuk memudahkan filtering
    prefixed_message = f"[{MODULE_LOGGER_NAME}] {message}"
    
    # Log ke Python logger
    if logger:
        if level == "info":
            logger.info(prefixed_message)
        elif level == "warning" or level == "warn":
            logger.warning(prefixed_message)
        elif level == "error":
            logger.error(prefixed_message)
        elif level == "debug":
            logger.debug(prefixed_message)
        elif level == "success":
            # Success level tidak ada di Python logger standard, gunakan info
            logger.info(f"✅ {prefixed_message}")
        elif level == "critical":
            logger.critical(prefixed_message)

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
        ui_components['logger'] = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    # Tambahkan fungsi log_message ke UI components
    ui_components['log_message'] = lambda msg, level="info", icon=None: log_message(ui_components, msg, level, icon)
    
    # Tambahkan flag download_initialized untuk menunjukkan modul telah di-initialize
    # Flag ini digunakan untuk mencegah log ke installer dependency
    ui_components['download_initialized'] = True
    
    # Tambahkan namespace ke ui_components untuk memudahkan tracing
    ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
    
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