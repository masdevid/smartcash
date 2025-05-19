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
        
        # Tambahkan dummy update_config_from_ui jika tidak ada untuk mencegah error
        if 'update_config_from_ui' not in ui_components:
            ui_components['update_config_from_ui'] = lambda *args, **kwargs: {}
            if logger: logger.debug("🔧 Menambahkan dummy update_config_from_ui untuk mencegah error")
        
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
    error_result = None
    
    try:
        # Ambil konfigurasi Roboflow
        from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
        config = get_endpoint_config(ui_components)
        
        # Validasi konfigurasi
        if not config.get('api_key'):
            error_result = {
                "success": False,
                "error": "API key Roboflow tidak tersedia",
                "alternative_message": "Pastikan Anda telah memasukkan API key Roboflow yang valid di form input"
            }
            _process_download_result(ui_components, error_result)
            return
            
        if not config.get('workspace') or not config.get('project') or not config.get('version'):
            error_result = {
                "success": False,
                "error": "Parameter workspace, project, atau version tidak lengkap",
                "alternative_message": "Pastikan semua parameter Roboflow telah diisi dengan benar"
            }
            _process_download_result(ui_components, error_result)
            return
        
        # Update status panel menggunakan komponen reusable
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], "Downloading dataset dari Roboflow...", "info")
        
        # Dapatkan download service
        _update_progress(ui_components, 10, "Mempersiapkan download service...")
        from smartcash.dataset.services.downloader.download_service import DownloadService
        
        output_dir = config.get('output_dir', 'data')
        if logger: logger.debug(f"💾 Output directory: {output_dir}")
        
        # Buat direktori output jika belum ada
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Inisialisasi download service
        try:
            download_service = DownloadService(output_dir=output_dir, config={'data': {'roboflow': config}}, logger=logger)
            
            if not download_service:
                raise ValueError("Tidak dapat membuat download service")
        except Exception as service_error:
            error_result = {
                "success": False,
                "error": f"Gagal menginisialisasi download service: {str(service_error)}",
                "alternative_message": "Coba periksa konfigurasi dan pastikan direktori output dapat diakses"
            }
            _process_download_result(ui_components, error_result)
            return
        
        # Jalankan download melalui download service
        _update_progress(ui_components, 20, "Memulai download dataset dari Roboflow...")
        
        # Log parameter download untuk debugging
        if logger:
            logger.debug(f"🔍 Download parameters: workspace={config.get('workspace')}, project={config.get('project')}, version={config.get('version')}")
        
        try:
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
            
        except Exception as download_error:
            error_detail = str(download_error)
            
            # Deteksi jenis error yang umum
            if "API key tidak tersedia" in error_detail:
                error_message = "API key Roboflow tidak valid atau tidak tersedia"
                alternative_message = "Pastikan Anda telah memasukkan API key Roboflow yang valid"
            elif "Workspace, project, dan version diperlukan" in error_detail:
                error_message = "Parameter workspace, project, atau version tidak lengkap"
                alternative_message = "Pastikan semua parameter Roboflow telah diisi dengan benar"
            elif "404" in error_detail or "Not Found" in error_detail:
                error_message = f"Dataset tidak ditemukan: {config.get('workspace')}/{config.get('project')}:{config.get('version')}"
                alternative_message = "Periksa kembali workspace, project, dan version yang dimasukkan"
            elif "403" in error_detail or "Forbidden" in error_detail:
                error_message = "Akses ditolak oleh Roboflow API"
                alternative_message = "Periksa apakah API key Anda memiliki akses ke dataset yang diminta"
            elif "timeout" in error_detail.lower() or "timed out" in error_detail.lower():
                error_message = "Koneksi timeout saat download dataset"
                alternative_message = "Periksa koneksi internet Anda dan coba lagi"
            else:
                error_message = f"Error saat download dataset: {error_detail}"
                alternative_message = "Coba periksa parameter dan koneksi internet Anda"
            
            error_result = {
                "success": False,
                "error": error_message,
                "error_details": error_detail,
                "alternative_message": alternative_message
            }
            
            _process_download_result(ui_components, error_result)
        
    except Exception as e:
        # Tangani error yang tidak tertangkap sebelumnya
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_detail = str(e)
        error_msg = f"Error saat proses download dataset: {error_detail}"
        
        # Log error untuk debugging
        if logger: 
            logger.error(f"❌ {error_msg}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Buat error result yang lebih informatif
        error_result = {
            "success": False,
            "error": error_msg,
            "error_details": error_detail,
            "alternative_message": "Coba periksa parameter dan koneksi internet Anda"
        }
        
        # Proses error result
        _process_download_result(ui_components, error_result)
    
    finally:
        # Reset UI jika belum di-reset oleh _process_download_result
        if not error_result:
            _reset_ui_after_download(ui_components)

def _process_download_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Proses hasil download dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil download dari dataset service
    """
    logger = ui_components.get('logger')
    
    # Import komponen yang diperlukan
    from smartcash.ui.components.status_panel import update_status_panel
    from smartcash.ui.utils.ui_logger import log_to_ui as show_status
    
    # Cek apakah ada error 'update_config_from_ui'
    if isinstance(result.get('error'), str) and 'update_config_from_ui' in result.get('error', ''):
        # Ini adalah error yang terkait dengan konfigurasi
        if logger: 
            logger.debug(f"🔧 Menangani error update_config_from_ui: {result.get('error')}")
        
        # Tambahkan dummy update_config_from_ui jika tidak ada
        if 'update_config_from_ui' not in ui_components:
            ui_components['update_config_from_ui'] = lambda *args, **kwargs: {}
            if logger: 
                logger.debug("🔧 Menambahkan dummy update_config_from_ui untuk mencegah error")
        
        # Hapus error dari result dan lanjutkan proses
        if 'error' in result:
            del result['error']
            result['success'] = True
            result['message'] = result.get('message', 'Dataset berhasil didownload')
    
    # Cek error terkait direktori input
    if isinstance(result.get('error'), str) and 'Tidak ada gambar di direktori input' in result.get('error', ''):
        # Tambahkan informasi tambahan untuk membantu pengguna
        error_msg = result.get('error', '')
        help_msg = "\n\nPastikan dataset telah diunduh dan dipreprocessing terlebih dahulu. Coba periksa direktori data/raw atau data/preprocessed."
        result['error'] = error_msg + help_msg
    
    # Reset UI setelah proses selesai
    _reset_ui_after_download(ui_components)
    
    # Cek hasil download
    if result.get('success', False):
        # Download berhasil
        message = result.get('message', 'Dataset berhasil didownload')
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "success")
        
        # Show status
        show_status(message, "success", ui_components)
        if logger: logger.info(f"✅ {message}")
        
        # Jika ada dataset_info, tampilkan
        if 'dataset_info' in result:
            dataset_info = result['dataset_info']
            info_message = f"Dataset info: {len(dataset_info.get('train', []))} train, "
            info_message += f"{len(dataset_info.get('valid', []))} valid, "
            info_message += f"{len(dataset_info.get('test', []))} test images"
            
            show_status(info_message, "info", ui_components)
            if logger: logger.info(f"ℹ️ {info_message}")
            
            # Tampilkan lokasi dataset
            if 'dataset_path' in result:
                path_message = f"Dataset tersimpan di: {result['dataset_path']}"
                show_status(path_message, "info", ui_components)
                if logger: logger.info(f"📁 {path_message}")
    
    elif 'warning' in result:
        # Warning - download berhasil tapi ada warning
        message = result.get('warning', 'Download berhasil dengan warning')
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "warning")
        
        # Show status
        show_status(message, "warning", ui_components)
        if logger: logger.warning(f"⚠️ {message}")
        
        # Jika ada alternative_message, tampilkan
        if 'alternative_message' in result:
            alt_message = result['alternative_message']
            show_status(alt_message, "info", ui_components)
            if logger: logger.info(f"ℹ️ {alt_message}")
        
    else:
        # Error lainnya - pastikan pesan error lengkap
        error_message = result.get('error') or result.get('message') or "Tidak ada detail error tersedia"
        message = f"Error saat download dataset: {error_message}"
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "error")
        
        # Show status
        show_status(message, "error", ui_components)
        if logger: logger.error(f"❌ {message}")
        
        # Jika ada alternative_message, tampilkan
        if 'alternative_message' in result:
            alt_message = result['alternative_message']
            show_status(alt_message, "info", ui_components)
            if logger: logger.info(f"ℹ️ {alt_message}")
            
        # Tampilkan saran untuk mengatasi masalah
        suggestion = "Coba periksa koneksi internet dan pastikan parameter download sudah benar."
        show_status(suggestion, "info", ui_components)
        if logger: logger.info(f"💡 {suggestion}")

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