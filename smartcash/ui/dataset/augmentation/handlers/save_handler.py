"""
File: smartcash/ui/dataset/augmentation/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi augmentasi dengan sinkronisasi Google Drive yang diperbaiki
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.augmentation.handlers.config_handler import save_augmentation_config

def handle_save_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol save konfigurasi augmentasi dengan Google Drive sync.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    ui_components = setup_ui_logger(ui_components)
    
    # Disable tombol selama proses
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        log_message(ui_components, "💾 Menyimpan konfigurasi augmentasi...", "info")
        
        # Simpan konfigurasi dengan sync ke Google Drive
        result = save_augmentation_config(ui_components)
        
        if result:
            log_message(ui_components, "✅ Konfigurasi berhasil disimpan dan disinkronkan ke Google Drive", "success")
            update_status_panel(ui_components, "✅ Konfigurasi tersimpan di Google Drive", "success")
        else:
            log_message(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
            update_status_panel(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
            
    except Exception as e:
        log_message(ui_components, f"❌ Error saat menyimpan: {str(e)}", "error")
        update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
    finally:
        if button and hasattr(button, 'disabled'):
            button.disabled = False