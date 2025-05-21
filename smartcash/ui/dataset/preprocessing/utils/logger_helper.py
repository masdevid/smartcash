"""
File: smartcash/ui/dataset/preprocessing/utils/logger_helper.py
Deskripsi: Helper untuk logging di preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.dataset.preprocessing.utils.notification_manager import PREPROCESSING_LOGGER_NAMESPACE

def setup_ui_logger(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup logger untuk UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Cek apakah logger sudah disetup
    if 'logger' in ui_components and 'log_message' in ui_components:
        return ui_components
    
    # Setup logger dengan namespace spesifik
    logger = create_ui_logger(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
    ui_components['logger'] = logger
    ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
    
    # Tambahkan helper function untuk log message
    ui_components['log_message'] = lambda message, level="info", icon="ℹ️": log_message(ui_components, message, level, icon)
    
    return ui_components

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = "ℹ️") -> None:
    """
    Log message ke UI dan logger.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan dilog
        level: Level log (info, warning, error, success)
        icon: Icon untuk pesan log
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Setup default logger jika tidak ada
    if 'logger' not in ui_components:
        ui_components['logger'] = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Log ke logger
    logger = ui_components['logger']
    if hasattr(logger, level):
        log_func = getattr(logger, level)
        log_func(message)
    else:
        logger.info(message)
    
    # Tampilkan di UI jika log_output tersedia
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_stdout'):
        formatted_message = f"{icon} {message}"
        
        # Gunakan stderr untuk level error
        if level.lower() == 'error':
            ui_components['log_output'].append_stderr(formatted_message)
        else:
            ui_components['log_output'].append_stdout(formatted_message)
    
    # Notifikasi melalui observer jika tersedia
    try:
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import notify_log
        notify_log(ui_components, message, level, icon)
    except (ImportError, AttributeError):
        pass

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika UI sudah diinisialisasi, False jika belum
    """
    # Periksa apakah memiliki flag initialized
    if 'preprocessing_initialized' in ui_components:
        return ui_components['preprocessing_initialized']
    
    # Periksa apakah memiliki logger dan UI components utama
    has_logger = 'logger' in ui_components
    has_ui = 'ui' in ui_components
    has_progress = 'progress_bar' in ui_components
    
    return has_logger and has_ui and has_progress 