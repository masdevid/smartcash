"""
File: smartcash/ui/dataset/split/handlers/sync_logger.py
Deskripsi: Utility untuk logging proses sinkronisasi konfigurasi split dataset ke UI logger
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def log_sync_status(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Log status sinkronisasi ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan (success, error, info, warning)
    """
    if 'logger' not in ui_components:
        return
    
    # Dapatkan icon berdasarkan status
    icon = ICONS.get(status, ICONS.get('info', 'ℹ️'))
    
    # Format pesan dengan icon
    formatted_message = f"{icon} {message}"
    
    # Log ke UI logger berdasarkan status
    if status == 'error':
        ui_components['logger'].error(formatted_message)
    elif status == 'warning':
        ui_components['logger'].warning(formatted_message)
    elif status == 'success':
        ui_components['logger'].info(formatted_message)
    else:
        ui_components['logger'].info(formatted_message)
    
    # Log juga ke console logger untuk keperluan debug
    if status == 'error':
        logger.error(formatted_message)
    elif status == 'warning':
        logger.warning(formatted_message)
    elif status == 'success':
        logger.info(formatted_message)
    else:
        logger.info(formatted_message)

def log_sync_success(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi sukses ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'success')

def log_sync_error(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi error ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'error')

def log_sync_warning(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi warning ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'warning')

def log_sync_info(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi info ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'info') 