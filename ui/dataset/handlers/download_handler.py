"""
File: smartcash/ui/dataset/handlers/download_handler.py
Deskripsi: Handler teroptimasi untuk download dataset dengan validasi yang konsisten
"""

from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor

def handle_download_button_click(b, ui_components: Dict[str, Any]) -> None:
    """Handle klik tombol download dengan validasi dan konfirmasi."""
    from smartcash.ui.utils.fallback_utils import show_status
    from smartcash.ui.dataset.handlers.confirmation_handler import confirm_download
    
    # Validasi berdasarkan endpoint yang dipilih (Roboflow/Drive/URL)
    endpoint, is_valid, message = ui_components['endpoint_dropdown'].value, False, ""
    
    if endpoint == 'Roboflow':
        is_valid, message = _validate_roboflow_input(ui_components)
    elif endpoint == 'Google Drive':
        is_valid, message = _validate_drive_input(ui_components)
    elif endpoint == 'URL Kustom':
        is_valid, message = _validate_url_input(ui_components)
    
    # Tampilkan pesan validasi jika gagal dan returni
    if not is_valid: 
        show_status(message, "error", ui_components)
        if logger := ui_components.get('logger'): logger.warning(f"⚠️ {message}")
        return
    
    # Lanjut ke konfirmasi jika valid
    confirm_download(ui_components, endpoint, b)

def _validate_roboflow_input(ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """Validasi input Roboflow."""
    # Dapatkan nilai dari input fields
    workspace = ui_components['rf_workspace'].value.strip()
    project = ui_components['rf_project'].value.strip()
    version = ui_components['rf_version'].value.strip()
    api_key = ui_components['rf_apikey'].value.strip()
    
    # Validasi masing-masing field
    if not workspace: return False, "Workspace ID tidak boleh kosong"
    if not project: return False, "Project ID tidak boleh kosong"
    if not version: return False, "Version tidak boleh kosong"
    if not api_key: return False, "API key tidak boleh kosong"
    if len(api_key) < 10: return False, "API key tidak valid (terlalu pendek)"
    
    return True, "Validasi berhasil"

def _validate_drive_input(ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """Validasi input Google Drive."""
    # TODO: Implementasi validasi Drive
    return True, "Google Drive dipilih"

def _validate_url_input(ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """Validasi input URL."""
    # TODO: Implementasi validasi URL
    return True, "URL kustom dipilih"

def execute_download(ui_components: Dict[str, Any], endpoint: str) -> None:
    """Eksekusi download dataset berdasarkan endpoint."""
    # Persiapkan UI untuk proses download
    _prepare_ui_for_download(ui_components, endpoint)
    
    # Jalankan download secara asinkron berdasarkan endpoint
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            if endpoint == 'Roboflow': executor.submit(download_from_roboflow, ui_components)
            elif endpoint == 'Google Drive': executor.submit(download_from_drive, ui_components)
            elif endpoint == 'URL Kustom': executor.submit(download_from_url, ui_components)
    except Exception as e:
        # Tangani error dan reset UI
        if logger := ui_components.get('logger'): logger.error(f"❌ Error saat memulai download: {str(e)}")
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"Error saat memulai download: {str(e)}", "error", ui_components)
        _reset_ui_after_download(ui_components, is_error=True)

def _prepare_ui_for_download(ui_components: Dict[str, Any], endpoint: str) -> None:
    """Persiapkan UI untuk proses download."""
    # Tampilkan progress bar dan reset nilainya
    ui_components['progress_bar'].layout.visibility = 'visible'
    ui_components['progress_message'].layout.visibility = 'visible'
    ui_components['progress_bar'].value = 0
    ui_components['progress_message'].value = f"Memulai download dataset dari {endpoint}..."
    
    # Update status panel
    from smartcash.ui.utils.constants import ALERT_STYLES
    ui_components['status_panel'].value = f"""
    <div style="padding:10px; background-color:{ALERT_STYLES['info']['bg_color']}; 
               color:{ALERT_STYLES['info']['text_color']}; border-radius:4px; margin:5px 0;
               border-left:4px solid {ALERT_STYLES['info']['text_color']};">
        <p style="margin:5px 0">{ALERT_STYLES['info']['icon']} Proses download dimulai...</p>
    </div>
    """
    
    # Log dimulainya download
    if logger := ui_components.get('logger'): logger.info(f"🚀 Memulai download dataset dari {endpoint}")

def _reset_ui_after_download(ui_components: Dict[str, Any], is_error: bool = False) -> None:
    """Reset UI setelah proses download selesai."""
    # Sembunyikan progress bar
    ui_components['progress_bar'].layout.visibility = 'hidden'
    ui_components['progress_message'].layout.visibility = 'hidden'

def download_from_roboflow(ui_components: Dict[str, Any]) -> None:
    """Download dataset dari Roboflow."""
    from smartcash.ui.utils.fallback_utils import show_status, get_dataset_manager
    from smartcash.ui.utils.constants import ALERT_STYLES
    
    logger = ui_components.get('logger')
    
    # Ambil parameter dari UI components
    workspace, project, version = ui_components['rf_workspace'].value, ui_components['rf_project'].value, ui_components['rf_version'].value
    api_key, output_format, output_dir = ui_components['rf_apikey'].value, ui_components['output_format'].value.lower().replace(' ', ''), ui_components['output_dir'].value
    
    # Set API key sebagai environment variable
    os.environ['ROBOFLOW_API_KEY'] = api_key
    
    try:
        # Dapatkan dataset manager dengan konfigurasi yang tepat
        download_service = get_dataset_manager({
            'data': {
                'dir': output_dir,
                'roboflow': {'workspace': workspace, 'project': project, 'version': version, 'api_key': api_key}
            }
        }, logger)
        
        if not download_service: raise Exception("Tidak dapat membuat service dataset")
            
        # Pull dataset (download dan export ke struktur lokal)
        result = download_service.pull_dataset(
            format=output_format, workspace=workspace, project=project, version=version, 
            api_key=api_key, show_progress=True, force_download=True, backup_existing=True
        )
        
        # Update UI berdasarkan hasil
        if result.get('status') in ['downloaded', 'local']:
            show_status(f"Download dataset berhasil! {result.get('stats', {}).get('total_images', 0)} gambar siap digunakan.", "success", ui_components)
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{ALERT_STYLES['success']['bg_color']}; 
                       color:{ALERT_STYLES['success']['text_color']}; border-radius:4px; margin:5px 0;
                       border-left:4px solid {ALERT_STYLES['success']['text_color']};">
                <p style="margin:5px 0">{ALERT_STYLES['success']['icon']} Dataset berhasil didownload dan disiapkan!</p>
            </div>
            """
    except Exception as e:
        # Tangani error download
        error_msg = f"Error saat download dataset: {str(e)}"
        if logger: logger.error(f"❌ {error_msg}")
        show_status(error_msg, "error", ui_components)
        ui_components['status_panel'].value = f"""
        <div style="padding:10px; background-color:{ALERT_STYLES['error']['bg_color']}; 
                   color:{ALERT_STYLES['error']['text_color']}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {ALERT_STYLES['error']['text_color']};">
            <p style="margin:5px 0">{ALERT_STYLES['error']['icon']} {error_msg}</p>
        </div>
        """
    finally:
        # Reset UI setelah selesai
        _reset_ui_after_download(ui_components)

def download_from_drive(ui_components: Dict[str, Any]) -> None:
    """Download dataset dari Google Drive."""
    if logger := ui_components.get('logger'): logger.info("ℹ️ Download dari Google Drive belum diimplementasikan")
    _reset_ui_after_download(ui_components)
    
def download_from_url(ui_components: Dict[str, Any]) -> None:
    """Download dataset dari URL Kustom."""
    if logger := ui_components.get('logger'): logger.info("ℹ️ Download dari URL kustom belum diimplementasikan")
    _reset_ui_after_download(ui_components)