"""
File: smartcash/ui/dataset/augmentation/handlers/reset_handler.py
Deskripsi: Handler untuk reset konfigurasi augmentasi dengan sinkronisasi Google Drive yang diperbaiki
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar
from smartcash.ui.dataset.augmentation.handlers.config_handler import reset_augmentation_config

def handle_reset_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol reset augmentasi dengan Google Drive sync.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    ui_components = setup_ui_logger(ui_components)
    
    # Disable tombol selama proses
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        update_status_panel(ui_components, "🔄 Mereset konfigurasi ke default...", "warning")
        log_message(ui_components, "🔄 Mereset konfigurasi augmentasi ke default...", "info")
        
        # Reset konfigurasi dengan save ke Google Drive
        result = reset_augmentation_config(ui_components)
        
        if result:
            # Reset status konfirmasi dan running flags
            _reset_ui_states(ui_components)
            
            # Reset progress bar
            reset_progress_bar(ui_components)
            
            log_message(ui_components, "✅ Konfigurasi berhasil direset dan disimpan ke Google Drive", "success")
            update_status_panel(ui_components, "✅ Konfigurasi direset dan tersinkronisasi", "success")
        else:
            log_message(ui_components, "❌ Gagal mereset konfigurasi", "error")
            update_status_panel(ui_components, "❌ Gagal mereset konfigurasi", "error")
        
    except Exception as e:
        log_message(ui_components, f"❌ Error saat mereset: {str(e)}", "error")
        update_status_panel(ui_components, f"❌ Gagal mereset: {str(e)}", "error")
    finally:
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def _reset_ui_states(ui_components: Dict[str, Any]) -> None:
    """Reset UI states dan flags internal."""
    
    # Reset status konfirmasi
    if 'confirmation_result' in ui_components:
        ui_components['confirmation_result'] = False
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
    
    # Reset flag running
    ui_components['augmentation_running'] = False
    ui_components['stop_requested'] = False
    
    log_message(ui_components, "🔄 UI states berhasil direset", "debug")