"""
File: smartcash/ui/training_config/hyperparameters/handlers/sync_logger.py
Deskripsi: Utilitas untuk mencatat status sinkronisasi konfigurasi hyperparameters di UI
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.training_config.hyperparameters.handlers.status_handlers import update_status_panel as base_update_status_panel

logger = get_logger(__name__)

def update_sync_status(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Update panel status sinkronisasi dengan pesan terbaru.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan (success, error, info, warning)
    """
    # Gunakan status_handlers yang sudah ada
    base_update_status_panel(ui_components, message, status)

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
    
    # Update status panel
    update_sync_status(ui_components, message, status)
    
    # Log ke UI logger berdasarkan status (tanpa emoji karena sudah ditambahkan oleh UILogger)
    if status == 'error':
        ui_components['logger'].error(message)
    elif status == 'warning':
        ui_components['logger'].warning(message)
    elif status == 'success':
        ui_components['logger'].success(message)  # Gunakan success method yang sudah ada
    else:
        ui_components['logger'].info(message)
    
    # Log juga ke console logger untuk keperluan debug (tanpa emoji)
    if status == 'error':
        logger.error(message)
    elif status == 'warning':
        logger.warning(message)
    elif status == 'success':
        logger.info(f"SUCCESS: {message}")
    else:
        logger.info(message)

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

def update_sync_status_only(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update panel status sinkronisasi di UI tanpa menambahkan log baru.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    try:
        # Gunakan status_handlers yang sudah ada untuk update panel
        base_update_status_panel(ui_components, message, status_type)
    except Exception as e:
        # Fallback ke logger jika ada error
        if 'logger' in ui_components:
            ui_components['logger'].error(f"Error saat update status panel: {str(e)}")
        logger.error(f"Error saat update status panel: {str(e)}") 