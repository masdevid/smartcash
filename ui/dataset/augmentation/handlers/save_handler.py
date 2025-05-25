"""
File: smartcash/ui/dataset/augmentation/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi augmentasi dengan button state manager terbaru
"""

from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.augmentation.handlers.config_handler import save_augmentation_config

def handle_save_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol save konfigurasi augmentasi dengan button state manager.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    ui_logger = create_ui_logger_bridge(ui_components, "save_handler")
    button_state_manager = get_button_state_manager(ui_components)
    
    # Cek apakah operation bisa dimulai
    can_start, reason = button_state_manager.can_start_operation("save_config")
    if not can_start:
        ui_logger.warning(f"⚠️ {reason}")
        return
    
    # Gunakan context manager untuk disable buttons selama save
    with button_state_manager.operation_context("save_config"):
        try:
            ui_logger.info("💾 Menyimpan konfigurasi augmentasi...")
            
            # Simpan konfigurasi dengan sync ke Google Drive
            result = save_augmentation_config(ui_components)
            
            if result:
                ui_logger.success("✅ Konfigurasi berhasil disimpan dan disinkronkan ke Google Drive")
                _update_status_panel(ui_components, "✅ Konfigurasi tersimpan di Google Drive", "success")
            else:
                ui_logger.error("❌ Gagal menyimpan konfigurasi")
                _update_status_panel(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
                
        except Exception as e:
            ui_logger.error(f"❌ Error saat menyimpan: {str(e)}")
            _update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")

def _update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel jika tersedia."""
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
        except ImportError:
            pass