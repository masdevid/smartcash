"""
File: smartcash/ui/dataset/download/handlers/confirmation_handler.py
Deskripsi: Handler konfirmasi untuk download dataset
"""

from typing import Dict, Any, Callable, Optional
from IPython.display import display

def confirm_download(ui_components: Dict[str, Any], endpoint: str, download_button) -> None:
    """Tampilkan dialog konfirmasi untuk download dataset."""
    # Import modul yang diperlukan
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    from smartcash.ui.utils.constants import ICONS, ALERT_STYLES
    from smartcash.ui.dataset.download.handlers.download_handler import execute_download
    
    logger = ui_components.get('logger')
    output_dir = ui_components['output_dir'].value
    
    # Buat pesan konfirmasi berdasarkan endpoint
    message = f"Anda akan mengunduh dataset dalam format YOLO v5 ke direktori {output_dir}. "
    message += _get_endpoint_details(ui_components, endpoint)
    
    # Fungsi untuk menjalankan download dan membersihkan dialog
    def confirm_and_execute():
        ui_components['confirmation_area'].clear_output()
        execute_download(ui_components, endpoint)
    
    # Tampilkan dialog konfirmasi
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title="Konfirmasi Download Dataset", message=message,
            on_confirm=confirm_and_execute,
            on_cancel=lambda: cancel_download(ui_components, logger)
        )
        display(dialog)
    
    # Update status panel
    ui_components['status_panel'].value = f"""
    <div style="padding:10px; background-color:{ALERT_STYLES['warning']['bg_color']}; 
               color:{ALERT_STYLES['warning']['text_color']}; border-radius:4px; margin:5px 0;
               border-left:4px solid {ALERT_STYLES['warning']['text_color']};">
        <p style="margin:5px 0">{ALERT_STYLES['warning']['icon']} Silakan konfirmasi untuk melanjutkan download dataset</p>
    </div>
    """
    
    if logger: logger.info(f"ℹ️ Menunggu konfirmasi download dataset dari {endpoint}")

def _get_endpoint_details(ui_components: Dict[str, Any], endpoint: str) -> str:
    """Dapatkan detail spesifik untuk setiap endpoint."""
    if endpoint == 'Roboflow':
        workspace = ui_components['rf_workspace'].value
        project = ui_components['rf_project'].value
        version = ui_components['rf_version'].value
        return f"Dataset akan diunduh dari Roboflow (workspace: {workspace}, project: {project}, version: {version})."
    elif endpoint == 'Google Drive':
        drive_folder = ui_components['drive_folder'].value
        return f"Dataset akan disinkronkan dari Google Drive folder: {drive_folder}."
    return ""

def cancel_download(ui_components: Dict[str, Any], logger=None) -> None:
    """Cancel download dan reset UI."""
    from smartcash.ui.utils.constants import ALERT_STYLES
    
    # Clear konfirmasi area
    ui_components['confirmation_area'].clear_output()
    
    # Update status panel
    ui_components['status_panel'].value = f"""
    <div style="padding:10px; background-color:{ALERT_STYLES['info']['bg_color']}; 
               color:{ALERT_STYLES['info']['text_color']}; border-radius:4px; margin:5px 0;
               border-left:4px solid {ALERT_STYLES['info']['text_color']};">
        <p style="margin:5px 0">{ALERT_STYLES['info']['icon']} Download dibatalkan</p>
    </div>
    """
    
    if logger: logger.info("ℹ️ Download dataset dibatalkan")