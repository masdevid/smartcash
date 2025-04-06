"""
File: smartcash/ui/dataset/download/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dengan integrasi langsung ke service
"""

from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

def handle_download_button_click(b, ui_components: Dict[str, Any]) -> None:
    """Handle klik tombol download dengan validasi dan konfirmasi."""
    from smartcash.ui.utils.fallback_utils import show_status
    from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download
    
    # Validasi berdasarkan endpoint yang dipilih
    endpoint, is_valid, message = ui_components['endpoint_dropdown'].value, False, ""
    
    if endpoint == 'Roboflow':
        is_valid, message = _validate_roboflow_input(ui_components)
    elif endpoint == 'Google Drive':
        is_valid, message = _validate_drive_input(ui_components)
    
    # Tampilkan pesan validasi jika gagal dan return
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
    # Validasi drive folder
    drive_folder = ui_components['drive_folder'].value.strip()
    if not drive_folder: return False, "Folder Google Drive tidak boleh kosong"
    
    # Validasi apakah drive terpasang
    try:
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        if not env.is_drive_mounted: return False, "Google Drive tidak terpasang"
    except ImportError:
        if not os.path.exists('/content/drive/MyDrive'): return False, "Google Drive tidak terpasang"
    
    return True, "Validasi berhasil"

def execute_download(ui_components: Dict[str, Any], endpoint: str) -> None:
    """Eksekusi download dataset berdasarkan endpoint."""
    # Persiapkan UI untuk proses download
    _prepare_ui_for_download(ui_components, endpoint)
    
    # Jalankan download secara asinkron berdasarkan endpoint
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            if endpoint == 'Roboflow': executor.submit(download_from_roboflow, ui_components)
            elif endpoint == 'Google Drive': executor.submit(download_from_drive, ui_components)
    except Exception as e:
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
    """Download dataset dari Roboflow menggunakan service."""
    from smartcash.ui.utils.fallback_utils import show_status
    from smartcash.ui.utils.constants import ALERT_STYLES
    
    logger = ui_components.get('logger')
    
    # Ambil parameter dari UI components
    workspace = ui_components['rf_workspace'].value
    project = ui_components['rf_project'].value
    version = ui_components['rf_version'].value
    api_key = ui_components['rf_apikey'].value
    output_dir = ui_components['output_dir'].value
    
    # Format selalu yolov5pytorch (tetap)
    output_format = "yolov5pytorch"
    
    # Set API key sebagai environment variable
    os.environ['ROBOFLOW_API_KEY'] = api_key
    
    try:
        # Gunakan DownloadService langsung, bukan melalui DatasetManager
        from smartcash.dataset.services.downloader.download_service import DownloadService
        
        # Buat instance DownloadService dengan konfigurasi yang tepat
        download_service = DownloadService(
            output_dir=output_dir,
            config={
                'data': {
                    'dir': output_dir,
                    'roboflow': {
                        'workspace': workspace,
                        'project': project,
                        'version': version,
                        'api_key': api_key
                    }
                }
            },
            logger=logger
        )
        
        if not download_service:
            raise Exception("Tidak dapat membuat download service")
            
        # Pull dataset (download dan export ke struktur lokal)
        result = download_service.pull_dataset(
            format=output_format,
            workspace=workspace,
            project=project,
            version=version,
            api_key=api_key,
            show_progress=True,
            force_download=True,
            backup_existing=True
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
    from smartcash.ui.dataset.download.handlers.drive_handler import process_drive_download
    process_drive_download(ui_components)