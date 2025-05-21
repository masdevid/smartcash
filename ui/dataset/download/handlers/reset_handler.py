"""
File: smartcash/ui/dataset/download/handlers/reset_handler.py
Deskripsi: Handler untuk reset operasi UI download
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.download.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.download.utils.progress_manager import reset_progress_bar

def handle_reset_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol reset pada UI download.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang memicu event (opsional)
    """
    try:
        # Setup logger jika belum
        ui_components = setup_ui_logger(ui_components)
        
        # Log pesan reset
        log_message(ui_components, "Mereset form download dataset...", "info", "🔄")
        
        # Reset semua field input
        _reset_input_fields(ui_components)
        
        # Reset status konfirmasi jika ada
        if 'confirmation_result' in ui_components:
            ui_components['confirmation_result'] = False
        
        # Bersihkan area konfirmasi jika ada
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Reset flag download_running jika ada
        if 'download_running' in ui_components:
            ui_components['download_running'] = False
        
        # Update status panel
        update_status_panel(
            ui_components,
            "Form telah direset",
            "success"
        )
        
        # Log pesan sukses
        log_message(ui_components, "Form berhasil direset", "success", "✅")
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat mereset form: {str(e)}", "error", "❌")
        
        # Update status panel dengan error
        update_status_panel(
            ui_components,
            "Gagal mereset form",
            "error"
        )

def _reset_input_fields(ui_components: Dict[str, Any]) -> None:
    """
    Reset semua field input ke nilai default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Definisi field dan nilai defaultnya
    default_values = {
        'workspace': '',
        'project': '',
        'version': '',
        'output_dir': '',
        'backup_checkbox': False,
        'backup_dir': '',
        'validate_dataset': True
    }
    
    # Reset each field if it exists in ui_components
    for field, default_value in default_values.items():
        if field in ui_components and hasattr(ui_components[field], 'value'):
            # Reset field
            ui_components[field].value = default_value
            log_message(ui_components, f"Reset {field} ke default", "debug", "🔄")
    
    # Khusus untuk API key, hanya reset jika checkbox reset_api dicentang
    if 'api_key' in ui_components and 'reset_api_checkbox' in ui_components:
        if ui_components['reset_api_checkbox'].value:
            ui_components['api_key'].value = ''
            log_message(ui_components, "Reset API key ke default", "debug", "🔑")
    
    # Reset progress bar jika ada
    reset_progress_bar(ui_components)
