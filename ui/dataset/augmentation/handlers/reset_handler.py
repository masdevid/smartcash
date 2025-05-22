"""
File: smartcash/ui/dataset/augmentation/handlers/reset_handler.py
Deskripsi: Handler untuk reset konfigurasi augmentasi (tanpa move_to_preprocessed)
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar

def handle_reset_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol reset augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    try:
        # Setup logger jika belum
        ui_components = setup_ui_logger(ui_components)
        
        update_status_panel(ui_components, "Sedang mereset konfigurasi...", "warning")
        log_message(ui_components, "Mereset konfigurasi augmentasi...", "info", "🔄")
        
        # Reset semua field input
        _reset_input_fields(ui_components)
        
        # Reset status konfirmasi
        if 'confirmation_result' in ui_components:
            ui_components['confirmation_result'] = False
        
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components:
            ui_components['confirmation_area'].clear_output()
        
        # Reset flag running
        if 'augmentation_running' in ui_components:
            ui_components['augmentation_running'] = False
        
        update_status_panel(ui_components, "Konfigurasi telah direset ke nilai default", "success")
        log_message(ui_components, "Konfigurasi berhasil direset", "success", "✅")
        
    except Exception as e:
        log_message(ui_components, f"Error saat mereset: {str(e)}", "error", "❌")
        update_status_panel(ui_components, f"Gagal mereset: {str(e)}", "error")

def _reset_input_fields(ui_components: Dict[str, Any]) -> None:
    """Reset semua field input ke nilai default (tanpa move_to_preprocessed)."""
    default_values = {
        'num_variations': 2,
        'target_count': 1000,
        'output_prefix': 'aug',
        'balance_classes': False,
        'validate_results': True
    }
    
    for field, default_value in default_values.items():
        if field in ui_components and hasattr(ui_components[field], 'value'):
            ui_components[field].value = default_value
            log_message(ui_components, f"Reset {field} ke default", "debug", "🔄")
    
    # Reset jenis augmentasi
    if 'augmentation_types' in ui_components and hasattr(ui_components['augmentation_types'], 'value'):
        ui_components['augmentation_types'].value = ['combined']
    
    # Reset progress bar
    reset_progress_bar(ui_components)