"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk button events di split dataset - refactored dengan SRP
"""

from typing import Dict, Any, Callable

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.split.handlers.save_handler import handle_save_action
from smartcash.ui.dataset.split.handlers.reset_handler import handle_reset_action

logger = get_logger(__name__)


def setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """
    Setup event handlers untuk tombol save dan reset.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Setup save button handler
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(_create_save_handler(ui_components))
            
        # Setup reset button handler  
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(_create_reset_handler(ui_components))
            
        logger.debug("🔗 Button handlers berhasil dipasang")
        
    except Exception as e:
        logger.error(f"💥 Error setup button handlers: {str(e)}")


def _create_save_handler(ui_components: Dict[str, Any]) -> Callable:
    """Buat handler untuk tombol save."""
    def on_save_clicked(button):
        handle_save_action(ui_components)
    return on_save_clicked


def _create_reset_handler(ui_components: Dict[str, Any]) -> Callable:
    """Buat handler untuk tombol reset."""
    def on_reset_clicked(button):
        handle_reset_action(ui_components)
    return on_reset_clicked