"""
File: smartcash/ui/setup/dependency_installer/utils/logger_helper.py
Deskripsi: Helper untuk penggunaan UI logger yang konsisten di module dependency installer
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

# Import namespace konstanta
from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke UI dan logger Python dengan namespace khusus dependency installer.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional untuk ditampilkan di depan pesan
    """
    # Cek apakah ini adalah dependency installer yang sudah diinisialisasi
    if not is_initialized(ui_components):
        # Skip UI logging jika belum diinisialisasi untuk mencegah log muncul di modul lain
        return
    
    # Pastikan menggunakan logger dengan namespace yang tepat
    logger = ui_components.get('logger') or get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    # Log ke UI hanya jika log_output atau output tersedia
    if 'log_output' in ui_components or 'output' in ui_components:
        # Log ke UI dengan konsisten menggunakan UI logger
        ui_log(ui_components, message, level, icon)
    
    # Tambahkan prefix untuk memudahkan filtering
    prefixed_message = f"[DEP-INSTALLER] {message}"
    
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

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI dependency installer sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika sudah diinisialisasi, False jika belum
    """
    return ui_components.get('dependency_installer_initialized', False) 