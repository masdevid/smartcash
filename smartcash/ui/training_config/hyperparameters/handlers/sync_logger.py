"""
File: smartcash/ui/training_config/hyperparameters/handlers/sync_logger.py
Deskripsi: Utilitas untuk mencatat status sinkronisasi konfigurasi hyperparameters di UI
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.training_config.hyperparameters.handlers.status_handlers import update_status_panel

logger = get_logger(__name__)

def update_sync_status_only(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update panel status sinkronisasi tanpa log tambahan.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    try:
        # Gunakan status_handlers untuk update panel
        update_status_panel(ui_components, message, status_type)
    except Exception as e:
        # Fallback ke logger jika ada error
        if 'logger' in ui_components:
            ui_components['logger'].error(f"Error saat update status panel: {str(e)}")
        logger.error(f"Error saat update status panel: {str(e)}")

def log_sync_success(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi sukses.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_sync_status_only(ui_components, f"{ICONS.get('success', '✅')} {message}", 'success')
    logger.info(f"SYNC SUCCESS: {message}")

def log_sync_error(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi error.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_sync_status_only(ui_components, f"{ICONS.get('error', '❌')} {message}", 'error')
    logger.error(f"SYNC ERROR: {message}")

def log_sync_warning(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi warning.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_sync_status_only(ui_components, f"{ICONS.get('warning', '⚠️')} {message}", 'warning')
    logger.warning(f"SYNC WARNING: {message}")

def log_sync_info(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi info.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_sync_status_only(ui_components, f"{ICONS.get('info', 'ℹ️')} {message}", 'info')
    logger.info(f"SYNC INFO: {message}")