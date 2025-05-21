"""
File: smartcash/ui/dataset/preprocessing/handlers/button_handler.py
Deskripsi: Handler untuk interaksi tombol pada UI preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_state, update_status_panel, is_preprocessing_running
)
from smartcash.ui.dataset.preprocessing.handlers.config_handler import get_preprocessing_config_from_ui
from smartcash.ui.dataset.preprocessing.handlers.confirmation_handler import confirm_preprocessing

def handle_preprocessing_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol preprocessing dataset.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Cek apakah preprocessing sudah berjalan
        if is_preprocessing_running(ui_components):
            log_message(ui_components, "Preprocessing sudah berjalan", "warning", "⚠️")
            return
            
        # Get config dari UI
        config = get_preprocessing_config_from_ui(ui_components)
        
        # Log mulai preprocessing
        log_message(ui_components, "Memulai preprocessing dataset...", "info", "🔄")
        
        # Update UI state
        update_status_panel(ui_components, "warning", "Konfirmasi preprocessing dataset...")
        
        # Tampilkan dialog konfirmasi
        confirm_preprocessing(ui_components, config, button)
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat persiapan preprocessing: {error_message}")
        log_message(ui_components, f"Error saat persiapan preprocessing: {error_message}", "error", "❌")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False 