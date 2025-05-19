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
    from smartcash.ui.components.status_panel import update_status_panel
    from smartcash.ui.dataset.download.handlers.download_handler import execute_download
    
    # Buat context logger khusus untuk download
    logger = ui_components.get('logger')
    download_logger = None
    if logger and hasattr(logger, 'bind'):
        download_logger = logger.bind(context="download_only")
        ui_components['download_logger'] = download_logger
    else:
        download_logger = logger
    
    output_dir = ui_components['output_dir'].value
    
    # Buat pesan konfirmasi berdasarkan endpoint
    message = f"Anda akan mengunduh dataset dalam format YOLO v5 ke direktori {output_dir}. "
    message += _get_endpoint_details(ui_components, endpoint)
    
    # Ambil custom cancel callback jika tersedia
    cancel_callback = ui_components.get('cancel_download_callback', 
                                       lambda: cancel_download(ui_components, download_logger))
    
    # Fungsi untuk menjalankan download dan membersihkan dialog
    def confirm_and_execute():
        # Pastikan bahwa kita hanya mengeksekusi proses download dan tidak ada proses lain
        try:
            # Bersihkan area konfirmasi
            ui_components['confirmation_area'].clear_output()
            
            # Log bahwa kita akan mengeksekusi download
            if download_logger:
                download_logger.debug(f"🔍 Mengeksekusi download dari {endpoint} ke {output_dir}")
            
            # Eksekusi download dengan konteks yang jelas
            ui_components['current_operation'] = 'download'
            execute_download(ui_components, endpoint)
        except Exception as e:
            if download_logger:
                download_logger.error(f"❌ Error saat eksekusi download: {str(e)}")
            # Reset UI jika terjadi error
            cancel_download(ui_components, download_logger)
    
    # Tampilkan dialog konfirmasi
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title="Konfirmasi Download Dataset", message=message,
            on_confirm=confirm_and_execute,
            on_cancel=cancel_callback
        )
        display(dialog)
    
    # Update status panel menggunakan komponen reusable
    update_status_panel(
        ui_components['status_panel'],
        "Silakan konfirmasi untuk melanjutkan download dataset",
        "warning"
    )
    
    if download_logger: download_logger.info(f"ℹ️ Menunggu konfirmasi download dataset dari {endpoint}")

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
    from smartcash.ui.components.status_panel import update_status_panel
    
    # Gunakan download_logger jika tersedia
    download_logger = ui_components.get('download_logger') or logger
    
    # Clear konfirmasi area
    ui_components['confirmation_area'].clear_output()
    
    # Reset UI dengan benar - pastikan tombol terlihat
    _reset_download_ui(ui_components)
    
    # Hapus konteks operasi jika ada
    if 'current_operation' in ui_components:
        if download_logger:
            download_logger.debug(f"🔧 Membersihkan operasi: {ui_components.get('current_operation')}")
        ui_components.pop('current_operation', None)
    
    # Update status panel menggunakan komponen reusable
    update_status_panel(
        ui_components['status_panel'],
        "Download dibatalkan",
        "info"
    )
    
    if download_logger: download_logger.info("ℹ️ Download dataset dibatalkan")
    
def _reset_download_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI download ke kondisi awal."""
    # Aktifkan kembali tombol
    for button_key in ['download_button', 'check_button']:
        if button_key in ui_components and hasattr(ui_components[button_key], 'disabled'):
            ui_components[button_key].disabled = False
        
        # Pastikan tombol terlihat
        if button_key in ui_components and hasattr(ui_components[button_key], 'layout'):
            ui_components[button_key].layout.display = 'block'
    
    # Reset progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
        ui_components['progress_bar'].layout.visibility = 'hidden'
        ui_components['progress_bar'].value = 0
    
    if 'progress_message' in ui_components and hasattr(ui_components['progress_message'], 'layout'):
        ui_components['progress_message'].layout.visibility = 'hidden'
        ui_components['progress_message'].value = ""
    
    # Reset tracker jika ada
    for tracker_key in ['download_tracker', 'download_step_tracker']:
        if tracker_key in ui_components:
            tracker = ui_components[tracker_key]
            if hasattr(tracker, 'reset'):
                tracker.reset()
            if hasattr(tracker, 'current'):
                tracker.current = 0
            if hasattr(tracker, 'set_description'):
                tracker.set_description("")