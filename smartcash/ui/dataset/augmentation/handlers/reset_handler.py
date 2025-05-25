"""
File: smartcash/ui/dataset/augmentation/handlers/reset_handler.py
Deskripsi: Handler untuk reset konfigurasi augmentasi dengan button state manager terbaru
"""

from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.augmentation.handlers.config_handler import reset_augmentation_config

def handle_reset_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol reset augmentasi dengan button state manager.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    ui_logger = create_ui_logger_bridge(ui_components, "reset_handler")
    button_state_manager = get_button_state_manager(ui_components)
    
    # Cek apakah operation bisa dimulai
    can_start, reason = button_state_manager.can_start_operation("reset_config")
    if not can_start:
        ui_logger.warning(f"⚠️ {reason}")
        return
    
    # Gunakan context manager untuk disable buttons selama reset
    with button_state_manager.operation_context("reset_config"):
        try:
            ui_logger.info("🔄 Mereset konfigurasi ke default...")
            
            # Reset konfigurasi dengan save ke Google Drive
            result = reset_augmentation_config(ui_components)
            
            if result:
                # Reset UI states
                _reset_ui_states(ui_components, ui_logger)
                
                # Reset progress bar menggunakan shared component
                _reset_progress_components(ui_components)
                
                ui_logger.success("✅ Konfigurasi berhasil direset dan disimpan ke Google Drive")
                _update_status_panel(ui_components, "✅ Konfigurasi direset dan tersinkronisasi", "success")
            else:
                ui_logger.error("❌ Gagal mereset konfigurasi")
                _update_status_panel(ui_components, "❌ Gagal mereset konfigurasi", "error")
            
        except Exception as e:
            ui_logger.error(f"❌ Error saat mereset: {str(e)}")
            _update_status_panel(ui_components, f"❌ Gagal mereset: {str(e)}", "error")

def _reset_ui_states(ui_components: Dict[str, Any], ui_logger) -> None:
    """Reset UI states dan flags internal."""
    
    # Reset status konfirmasi
    if 'confirmation_result' in ui_components:
        ui_components['confirmation_result'] = False
    
    # Bersihkan area konfirmasi menggunakan shared component
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
    
    # Reset flag running
    ui_components['augmentation_running'] = False
    ui_components['stop_requested'] = False
    
    ui_logger.debug("🔄 UI states berhasil direset")

def _reset_progress_components(ui_components: Dict[str, Any]) -> None:
    """Reset komponen progress ke kondisi awal menggunakan shared component."""
    # Reset menggunakan shared progress tracking
    if 'reset_all' in ui_components and callable(ui_components['reset_all']):
        ui_components['reset_all']()
    elif 'tracker' in ui_components:
        ui_components['tracker'].reset()
    else:
        # Fallback manual reset
        if 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
            if hasattr(progress_bar, 'value'):
                progress_bar.value = 0
                progress_bar.description = "Progress: 0%"
            if hasattr(progress_bar, 'layout'):
                progress_bar.layout.visibility = 'hidden'
        
        for label_key in ['progress_message', 'step_label', 'overall_label']:
            if label_key in ui_components:
                label = ui_components[label_key]
                if hasattr(label, 'value'):
                    label.value = ""
                if hasattr(label, 'layout'):
                    label.layout.visibility = 'hidden'
        
        if 'progress_container' in ui_components:
            container = ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.display = 'none'

def _update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel jika tersedia."""
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
        except ImportError:
            pass