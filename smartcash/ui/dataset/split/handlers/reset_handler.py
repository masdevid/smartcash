"""
File: smartcash/ui/dataset/split/handlers/reset_handler.py
Deskripsi: Handler untuk aksi reset konfigurasi split dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
from smartcash.ui.dataset.split.handlers.ui_updater import update_ui_from_config
from smartcash.ui.dataset.split.handlers.status_updater import update_status

logger = get_logger(__name__)


def handle_reset_action(ui_components: Dict[str, Any]) -> None:
    """
    Handle aksi reset konfigurasi ke default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status(ui_components, "🔄 Mereset konfigurasi ke default...", 'info')
        
        # Get default config
        default_config = get_default_split_config()
        
        # Update UI dengan config default
        update_ui_from_config(ui_components, default_config)
        
        # Save default config
        config_manager = ui_components.get('config_manager')
        if config_manager:
            config_manager.save_config(default_config, 'dataset_config')
            
        update_status(ui_components, "✅ Konfigurasi berhasil direset ke default", 'success')
        logger.success("🔄 Konfigurasi split berhasil direset")
        
    except Exception as e:
        error_msg = f"Error saat reset konfigurasi: {str(e)}"
        update_status(ui_components, error_msg, 'error')
        logger.error(f"💥 {error_msg}")