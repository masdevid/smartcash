"""
File: smartcash/ui/dataset/split/handlers/reset_handlers.py
Deskripsi: Handler untuk reset konfigurasi di split dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.common.constants.log_messages import (
    OPERATION_SUCCESS, OPERATION_FAILED, CONFIG_LOADED
)
from smartcash.ui.dataset.split.handlers.config_handlers import load_config, update_ui_from_config
from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel

logger = get_logger(__name__)

def handle_reset_action(ui_components: Dict[str, Any]) -> None:
    """
    Handle aksi reset konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Memuat ulang konfigurasi...", 'info')
        
        # Load konfigurasi
        config = load_config()
        
        # Update UI dari konfigurasi
        update_ui_from_config(ui_components, config)
        
        # Update status
        update_status_panel(ui_components, "Konfigurasi berhasil dimuat ulang", 'success')
        logger.info(OPERATION_SUCCESS.format(operation="Reset konfigurasi"))
    except Exception as e:
        # Update status
        error_message = f"Error saat reset: {str(e)}"
        update_status_panel(ui_components, error_message, 'error')
        logger.error(OPERATION_FAILED.format(operation="Reset konfigurasi", reason=str(e)))

def create_reset_handler(ui_components: Dict[str, Any]):
    """
    Buat handler untuk tombol reset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Function handler untuk tombol reset
    """
    def on_reset_clicked(b):
        handle_reset_action(ui_components)
    
    return on_reset_clicked 