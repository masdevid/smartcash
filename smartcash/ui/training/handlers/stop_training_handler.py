"""
File: smartcash/ui/training/handlers/stop_training_handler.py
Deskripsi: Handler untuk stop training button
"""

from typing import Dict, Any
from smartcash.ui.training.handlers.training_button_handlers import get_state, set_state
from smartcash.ui.training.utils.training_status_utils import update_training_status


def handle_stop_training(ui_components: Dict[str, Any]):
    """Handle stop training dengan graceful checkpoint saving"""
    if not get_state()['active']:
        return
    
    set_state(stop_requested=True)
    ui_components['stop_button'].disabled = True
    update_training_status(ui_components, "⏹️ Menghentikan training dan menyimpan checkpoint...", 'warning')
    
    # Gunakan training manager jika tersedia
    training_manager = ui_components.get('training_manager')
    if training_manager:
        training_manager.stop_training()
    else:
        # Fallback ke training service langsung
        training_service = ui_components.get('training_service')
        if training_service:
            training_service.stop_training()
    
    logger = ui_components.get('logger')
    logger and logger.info("⏹️ Training stop requested dengan checkpoint saved")
