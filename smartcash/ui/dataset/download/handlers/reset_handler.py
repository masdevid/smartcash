"""
File: smartcash/ui/dataset/download/handlers/reset_handler.py
Deskripsi: Handler untuk reset UI dan state pada modul download
"""

from typing import Dict, Any, Optional

def handle_reset_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol reset pada UI download.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Reset UI components
        reset_download_ui(ui_components)
        
        # Log reset berhasil
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, "UI download berhasil direset", "info", "🔄")
        if logger: logger.info("🔄 UI download berhasil direset")
        
        # Update status panel jika tersedia
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel']("Konfigurasi download dataset", "info")
        
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat reset UI: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")

def reset_download_ui(ui_components: Dict[str, Any]) -> None:
    """
    Reset semua komponen UI download ke nilai default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset input fields
    if 'url_input' in ui_components:
        ui_components['url_input'].value = ''
    
    if 'dataset_type' in ui_components:
        ui_components['dataset_type'].value = 'currency'
    
    if 'save_path' in ui_components:
        ui_components['save_path'].value = 'data/raw'
    
    if 'auto_extract' in ui_components:
        ui_components['auto_extract'].value = True
    
    if 'validate_dataset' in ui_components:
        ui_components['validate_dataset'].value = True
    
    # Reset progress tracking
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar']()
    else:
        # Fallback ke reset progress bar manual
        _reset_progress_bar(ui_components)
    
    # Reset summary container
    if 'summary_container' in ui_components:
        ui_components['summary_container'].clear_output()
        ui_components['summary_container'].layout.display = 'none'
    
    # Reset log output
    if 'log_output' in ui_components:
        ui_components['log_output'].clear_output()
    
    # Aktifkan tombol-tombol
    _enable_buttons(ui_components)

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke nilai awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Reset labels
    for label_key in ['overall_label', 'step_label']:
        if label_key in ui_components:
            ui_components[label_key].value = ""
            ui_components[label_key].layout.visibility = 'hidden'
    
    # Reset current progress
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = 0
        ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Reset progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.visibility = 'hidden'

def _enable_buttons(ui_components: Dict[str, Any]) -> None:
    """
    Aktifkan semua tombol UI.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Daftar tombol yang perlu diaktifkan
    button_keys = ['download_button', 'check_button', 'reset_button']
    
    # Set status enabled untuk semua tombol
    for key in button_keys:
        if key in ui_components:
            ui_components[key].disabled = False
            if hasattr(ui_components[key], 'layout'):
                ui_components[key].layout.display = 'block'
