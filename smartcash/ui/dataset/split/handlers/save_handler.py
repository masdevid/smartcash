"""
File: smartcash/ui/dataset/split/handlers/save_handler.py
Deskripsi: Handler untuk aksi save konfigurasi split dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.ui_extractor import extract_ui_values
from smartcash.ui.dataset.split.handlers.config_updater import update_and_save_config
from smartcash.ui.dataset.split.handlers.status_updater import update_status

logger = get_logger(__name__)


def handle_save_action(ui_components: Dict[str, Any]) -> None:
    """
    Handle aksi save konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status(ui_components, "💾 Menyimpan konfigurasi split dataset...", 'info')
        
        # Extract nilai dari UI
        ui_values = extract_ui_values(ui_components)
        logger.debug(f"📊 Nilai UI: {ui_values}")
        
        # Update dan save config
        success, message = update_and_save_config(ui_components, ui_values)
        
        if success:
            update_status(ui_components, f"✅ {message}", 'success')
            logger.success(f"💾 {message}")
        else:
            update_status(ui_components, f"❌ {message}", 'error')
            logger.error(f"💥 {message}")
            
    except Exception as e:
        error_msg = f"Error saat menyimpan konfigurasi: {str(e)}"
        update_status(ui_components, error_msg, 'error')
        logger.error(f"💥 {error_msg}")