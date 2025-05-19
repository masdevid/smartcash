"""
File: smartcash/ui/dataset/download/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dengan dukungan observer dan delegasi ke service yang sesuai
"""

from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

def handle_download_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dataset.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # Nonaktifkan tombol selama proses
    _disable_buttons(ui_components, True)
    
    try:
        # Reset progress bar terlebih dahulu
        _reset_progress_bar(ui_components)
        
        # Dapatkan endpoint yang dipilih
        endpoint = ui_components.get('endpoint_dropdown', {}).value
        
        # Konfirmasi download
        from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download, cancel_download
        # Sebelum menampilkan konfirmasi, persiapkan cancel_callback
        def cancel_callback():
            # Pastikan tombol diaktifkan kembali saat cancel
            _disable_buttons(ui_components, False)
            cancel_download(ui_components, logger)
            
        # Tetapkan callback ke ui_components agar bisa diakses di confirmation_handler
        ui_components['cancel_download_callback'] = cancel_callback
        
        # Tampilkan konfirmasi
        confirm_download(ui_components, endpoint, b)
        
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat persiapan download: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
        
        # Aktifkan kembali tombol
        _disable_buttons(ui_components, False)

def execute_download(ui_components: Dict[str, Any], endpoint: str) -> None:
    """
    Eksekusi proses download dataset sesuai dengan endpoint yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Endpoint yang dipilih ('Roboflow' atau 'Google Drive')
    """
    logger = ui_components.get('logger')
    
    try:
        # Reset UI dan persiapan download
        _show_progress(ui_components, "Mempersiapkan download...")
        
        # Cek API key jika endpoint adalah Roboflow
        if endpoint == 'Roboflow':
            # Validasi API key
            from smartcash.ui.dataset.download.handlers.api_key_handler import check_api_key
            has_key, api_key = check_api_key(ui_components)
            
            if not has_key:
                # Tampilkan error
                from smartcash.ui.utils.ui_logger import log_to_ui
                error_msg = "API key Roboflow diperlukan untuk download"
                log_to_ui(ui_components, error_msg, "error", "🔑")
                if logger: logger.error(f"❌ {error_msg}")
                
                # Minta input API key
                from smartcash.ui.dataset.download.handlers.api_key_handler import request_api_key_input
                request_api_key_input(ui_components)
                
                # Reset UI
                _reset_ui_after_download(ui_components)
                return
            
            # Jalankan download dari Roboflow dalam thread terpisah
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(_download_from_roboflow, ui_components)
        
        elif endpoint == 'Google Drive':
            # Jalankan download dari Drive dalam thread terpisah
            from smartcash.ui.dataset.download.handlers.drive_handler import process_drive_download
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(process_drive_download, ui_components)
                
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat download dataset: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
        
        # Reset UI
        _reset_ui_after_download(ui_components)

def _download_from_roboflow(ui_components: Dict[str, Any]) -> None:
    """
    Download dataset dari Roboflow menggunakan dataset manager.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Ambil konfigurasi Roboflow
        from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
        config = get_endpoint_config(ui_components)
        
        # Update status panel menggunakan komponen reusable
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], "Downloading dataset dari Roboflow...", "info")
        
        # Dapatkan download service
        _update_progress(ui_components, 10, "Mempersiapkan download service...")
        from smartcash.dataset.services.downloader.download_service import DownloadService
        
        output_dir = config.get('output_dir', 'data')
        download_service = DownloadService(output_dir=output_dir, config={'data': {'roboflow': config}}, logger=logger)
        
        if not download_service:
            raise ValueError("Tidak dapat membuat download service")
        
        # Jalankan download melalui download service
        _update_progress(ui_components, 20, "Memulai download dataset dari Roboflow...")
        
        result = download_service.download_from_roboflow(
            api_key=config.get('api_key'),
            workspace=config.get('workspace'),
            project=config.get('project'), 
            version=config.get('version'),
            format=config.get('format', 'yolov5pytorch'),
            show_progress=True
        )
        
        # Analisis hasil
        _process_download_result(ui_components, result)
        
    except Exception as e:
        # Tampilkan error - gunakan string error yang spesifik
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat download dataset: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
        
        # Update status panel dengan error
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], f"Error saat download dataset: {str(e)}", "error")
    
    finally:
        # Reset UI
        _reset_ui_after_download(ui_components)

def _process_download_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Proses hasil download dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil download dari dataset service
    """
    logger = ui_components.get('logger')
    
    # Update Progress ke 100%
    _update_progress(ui_components, 100, "Download selesai")
    
    # Cek status hasil
    status = result.get('status')
    
    from smartcash.ui.utils.fallback_utils import show_status
    from smartcash.ui.components.status_panel import update_status_panel
    
    if status == 'success':
        # Sukses download
        stats = result.get('stats', {})
        message = f"Dataset berhasil didownload: {stats.get('total_images', 0)} gambar"
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "success")
        
        # Show status
        show_status(message, "success", ui_components)
        if logger: logger.success(f"✅ {message}")
        
    elif status == 'local':
        # Dataset sudah ada di lokal
        stats = result.get('stats', {})
        message = f"Dataset sudah tersedia di lokal: {stats.get('total_images', 0)} gambar"
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "info")
        
        # Show status
        show_status(message, "info", ui_components)
        if logger: logger.info(f"ℹ️ {message}")
        
    elif status == 'partial':
        # Dataset parsial/error
        stats = result.get('stats', {})
        message = f"Dataset tersedia sebagian: {stats.get('total_images', 0)} gambar. Error: {result.get('error')}"
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "warning")
        
        # Show status
        show_status(message, "warning", ui_components)
        if logger: logger.warning(f"⚠️ {message}")
        
    else:
        # Error lainnya - pastikan pesan error lengkap
        error_message = result.get('error') or result.get('message') or "Tidak ada detail error tersedia"
        message = f"Error saat download dataset: {error_message}"
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "error")
        
        # Show status
        show_status(message, "error", ui_components)
        if logger: logger.error(f"❌ {message}")

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke nilai awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = ""
        ui_components['progress_bar'].layout.visibility = 'hidden'
        ui_components['progress_message'].layout.visibility = 'hidden'
        
        # Reset tracker jika ada
        for tracker_key in ['download_tracker', 'download_step_tracker']:
            if tracker_key in ui_components:
                tracker = ui_components[tracker_key]
                if hasattr(tracker, 'reset'):
                    tracker.reset()
                tracker.current = 0
                if hasattr(tracker, 'set_description'):
                    tracker.set_description("")

def _show_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """
    Tampilkan dan reset progress bar.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan progress awal
    """
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = message
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'

def _update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """
    Update progress bar.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress opsional
    """
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = value
        
    if message and 'progress_message' in ui_components:
        ui_components['progress_message'].value = message
        
    # Update progress tracker jika tersedia
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)

def _disable_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Nonaktifkan/aktifkan tombol-tombol UI.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang perlu dinonaktifkan
    button_keys = ['download_button', 'check_button']
    
    # Set status disabled untuk semua tombol
    for key in button_keys:
        if key in ui_components:
            ui_components[key].disabled = disabled

def _reset_ui_after_download(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses download selesai."""
    # Aktifkan kembali tombol
    _disable_buttons(ui_components, False)
    
    # Pastikan tombol terlihat
    for button_key in ['download_button', 'check_button']:
        if button_key in ui_components and hasattr(ui_components[button_key], 'layout'):
            ui_components[button_key].layout.display = 'block'
    
    # Sembunyikan progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
        ui_components['progress_bar'].layout.visibility = 'hidden'
    if 'progress_message' in ui_components and hasattr(ui_components['progress_message'], 'layout'):
        ui_components['progress_message'].layout.visibility = 'hidden'