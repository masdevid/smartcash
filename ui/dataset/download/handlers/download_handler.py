"""
File: smartcash/ui/dataset/download/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dengan dukungan observer dan delegasi ke service yang sesuai
"""

from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import os
import time

def handle_download_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dataset dari Roboflow.
    
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
        confirm_download(ui_components, 'Roboflow', b)
        
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat persiapan download: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
        
        # Aktifkan kembali tombol
        _disable_buttons(ui_components, False)

def execute_download(ui_components: Dict[str, Any], endpoint: str = 'Roboflow') -> None:
    """
    Eksekusi proses download dataset dari Roboflow.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Parameter dipertahankan untuk kompatibilitas, selalu 'Roboflow'
    """
    logger = ui_components.get('logger')
    
    # Buat context logger khusus untuk download yang tidak mempengaruhi modul lain
    download_logger = logger
    
    # Coba gunakan bind jika tersedia, jika tidak gunakan logger biasa
    try:
        if logger and hasattr(logger, 'bind'):
            download_logger = logger.bind(context="download_only")
            ui_components['download_logger'] = download_logger
    except Exception as e:
        # Jika terjadi error saat bind, gunakan logger biasa
        if logger:
            # Log error untuk debugging (opsional)
            logger.debug(f"🔧 Tidak dapat membuat context logger: {str(e)}")
        # Simpan logger ke ui_components untuk digunakan nanti
        ui_components['download_logger'] = download_logger
    
    try:
        # Reset UI dan persiapan download
        _show_progress(ui_components, "Mempersiapkan download dari Roboflow...")
        
        # Tambahkan dummy update_config_from_ui jika tidak ada untuk mencegah error
        if 'update_config_from_ui' not in ui_components:
            ui_components['update_config_from_ui'] = lambda *args, **kwargs: {}
            if download_logger: 
                download_logger.debug("🔧 Menambahkan dummy update_config_from_ui untuk mencegah error")
        
        # Validasi API key
        from smartcash.ui.dataset.download.handlers.api_key_handler import check_api_key
        has_key, api_key = check_api_key(ui_components)
        
        if not has_key:
            # Tampilkan error
            from smartcash.ui.utils.ui_logger import log_to_ui
            error_msg = "API key Roboflow diperlukan untuk download"
            log_to_ui(ui_components, error_msg, "error", "🔑")
            
            # Reset UI
            _reset_ui_after_download(ui_components)
            return
            
        # Download dari Roboflow
        _download_from_roboflow(ui_components)
    
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
    # Gunakan download_logger jika tersedia, jika tidak gunakan logger biasa
    logger = ui_components.get('download_logger') or ui_components.get('logger')
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
                "alternative_message": "Pastikan Anda telah memasukkan API key Roboflow yang valid di form input",
                "context": "download_only"
            }
            _process_download_result(ui_components, error_result)
            return
            
        if not config.get('workspace') or not config.get('project') or not config.get('version'):
            error_result = {
                "success": False,
                "error": "Parameter workspace, project, atau version tidak lengkap",
                "alternative_message": "Pastikan semua parameter Roboflow telah diisi dengan benar",
                "context": "download_only"
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
                "alternative_message": "Coba periksa konfigurasi dan pastikan direktori output dapat diakses",
                "context": "download_only"
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
                "alternative_message": alternative_message,
                "context": "download_only"
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
            "alternative_message": "Coba periksa parameter dan koneksi internet Anda",
            "context": "download_only"
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
    
    # Buat context logger khusus untuk download yang tidak mempengaruhi modul lain
    # Gunakan download_logger yang sudah ada jika tersedia, jika tidak gunakan logger biasa
    download_logger = ui_components.get('download_logger') or logger
    
    # Jika download_logger belum ada, coba buat dengan bind jika tersedia
    if not ui_components.get('download_logger'):
        try:
            if logger and hasattr(logger, 'bind'):
                download_logger = logger.bind(context="download_only")
                ui_components['download_logger'] = download_logger
        except Exception as e:
            # Jika terjadi error saat bind, gunakan logger biasa
            download_logger = logger
    
    # Filter error yang tidak relevan dengan proses download
    if isinstance(result.get('error'), str):
        error_msg = result.get('error', '')
        
        # Daftar kata kunci untuk error yang harus diabaikan
        ignored_error_keywords = [
            'update_config_from_ui',           # Error konfigurasi
            'Tidak ada gambar di direktori input',  # Error validasi gambar
            'augmentasi',                     # Error augmentasi
            'augment',                        # Error augmentasi
            'pipeline',                       # Error pipeline augmentasi
            'split',                          # Error split augmentasi
            'worker',                         # Error worker augmentasi
            'factory'                         # Error factory augmentasi
        ]
        
        # Cek apakah error mengandung kata kunci yang harus diabaikan
        should_ignore = False
        matched_keyword = None
        for keyword in ignored_error_keywords:
            if keyword.lower() in error_msg.lower():
                should_ignore = True
                matched_keyword = keyword
                break
        
        # Jika error harus diabaikan
        if should_ignore:
            if download_logger: 
                download_logger.debug(f"🔧 Mengabaikan error yang tidak relevan dengan download (keyword: {matched_keyword}): {error_msg}")
            
            # Hapus error dari result dan lanjutkan proses
            if 'error' in result:
                del result['error']
                result['success'] = True
                result['message'] = result.get('message', 'Dataset berhasil didownload')
    
    # Reset UI setelah proses selesai
    _reset_ui_after_download(ui_components)
    
    # Pastikan flag prevent_augmentation dihapus setelah proses selesai
    ui_components.pop('prevent_augmentation', None)
    
    # Cek hasil download
    if result.get('success', False):
        # Download berhasil
        message = result.get('message', 'Dataset berhasil didownload')
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "success")
        
        # Show status
        show_status(message, "success", ui_components)
        
        # Gunakan download_logger jika tersedia, jika tidak gunakan logger biasa
        download_logger = ui_components.get('download_logger') or logger
        if download_logger: download_logger.info(f"✅ {message}")
        
        # Jika ada dataset_info, tampilkan
        if 'dataset_info' in result:
            dataset_info = result['dataset_info']
            info_message = f"Dataset info: {len(dataset_info.get('train', []))} train, "
            info_message += f"{len(dataset_info.get('valid', []))} valid, "
            info_message += f"{len(dataset_info.get('test', []))} test images"
            
            show_status(info_message, "info", ui_components)
            
            # Gunakan download_logger jika tersedia
            download_logger = ui_components.get('download_logger') or logger
            if download_logger: download_logger.info(f"ℹ️ {info_message}")
            
            # Tampilkan lokasi dataset
            if 'dataset_path' in result:
                path_message = f"Dataset tersimpan di: {result['dataset_path']}"
                show_status(path_message, "info", ui_components)
                if download_logger: download_logger.info(f"📁 {path_message}")
    
    elif 'warning' in result:
        # Warning - download berhasil tapi ada warning
        message = result.get('warning', 'Download berhasil dengan warning')
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "warning")
        
        # Show status
        show_status(message, "warning", ui_components)
        
        # Gunakan download_logger jika tersedia
        download_logger = ui_components.get('download_logger') or logger
        if download_logger: download_logger.warning(f"⚠️ {message}")
        
        # Jika ada alternative_message, tampilkan
        if 'alternative_message' in result:
            alt_message = result['alternative_message']
            show_status(alt_message, "info", ui_components)
            if download_logger: download_logger.info(f"ℹ️ {alt_message}")
        
    else:
        # Error lainnya - pastikan pesan error lengkap
        error_message = result.get('error') or result.get('message') or "Tidak ada detail error tersedia"
        message = f"Error saat download dataset: {error_message}"
        
        # Update status panel
        update_status_panel(ui_components['status_panel'], message, "error")
        
        # Daftar kata kunci untuk pesan yang harus diabaikan di UI
        augmentation_keywords = ['augmentasi', 'augment', 'pipeline', 'worker', 'split', 'factory', 'update_config_from_ui']
        
        # Cek apakah pesan error mengandung kata kunci yang harus diabaikan
        should_ignore_ui = any(keyword.lower() in message.lower() for keyword in augmentation_keywords)
        
        # Hanya tampilkan error jika konteksnya adalah download_only atau tidak ada konteks
        # dan tidak mengandung kata kunci yang harus diabaikan
        if (not result.get('context') or result.get('context') == 'download_only') and not should_ignore_ui:
            # Show status
            show_status(message, "error", ui_components)
            
            # Gunakan download_logger jika tersedia
            download_logger = ui_components.get('download_logger') or logger
            if download_logger: download_logger.error(f"❌ {message}")
            
            # Jika ada alternative_message, tampilkan
            if 'alternative_message' in result:
                alt_message = result['alternative_message']
                show_status(alt_message, "info", ui_components)
                if download_logger: download_logger.info(f"ℹ️ {alt_message}")
                
            # Tampilkan saran untuk mengatasi masalah
            suggestion = "Coba periksa koneksi internet dan pastikan parameter download sudah benar."
            show_status(suggestion, "info", ui_components)
            if download_logger: download_logger.info(f"💡 {suggestion}")
        else:
            # Jika konteks bukan download_only atau mengandung kata kunci yang harus diabaikan, hanya log untuk debugging
            if logger: 
                if should_ignore_ui:
                    logger.debug(f"🔧 Mengabaikan pesan augmentasi di UI: {message}")
                else:
                    logger.debug(f"🔧 Mengabaikan error dari konteks lain: {message}")

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke nilai awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Gunakan reset_progress dari shared component jika tersedia
    try:
        from smartcash.ui.components.progress_tracking import reset_progress
        reset_progress(ui_components)
        return
    except Exception as e:
        # Log error jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"⚠️ Error menggunakan shared progress reset: {str(e)}")
        
    # Reset progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Reset labels
    for label_key in ['overall_label', 'step_label', 'progress_message']:
        if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
            ui_components[label_key].value = ""
            ui_components[label_key].layout.visibility = 'hidden'
    
    # Reset tracker jika tersedia
    for tracker_key in ['download_tracker', 'download_step_tracker']:
        if tracker_key in ui_components:
            tracker = ui_components[tracker_key]
            if hasattr(tracker, 'current'):
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
    # Gunakan update_progress dari shared component jika tersedia
    try:
        from smartcash.ui.components.progress_tracking import update_progress
        
        update_progress(
            ui_components=ui_components,
            progress=0,
            total=100,
            message=message,
            step=0,
            total_steps=1,
            step_message=message,
            status_type='info'
        )
    except Exception as e:
        # Log error jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"⚠️ Error menggunakan shared progress tracking: {str(e)}")
            
        # Fallback ke implementasi sederhana
        if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].layout.visibility = 'visible'
            
        for label_key in ['overall_label', 'step_label', 'progress_message']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].value = message
                ui_components[label_key].layout.visibility = 'visible'

def _update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """
    Update progress bar.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress opsional
    """
    # Gunakan update_progress dari shared component jika tersedia
    try:
        from smartcash.ui.components.progress_tracking import update_progress
        
        update_progress(
            ui_components=ui_components,
            progress=value,
            total=100,
            message=message,
            status_type='info'
        )
    except Exception as e:
        # Log error jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"⚠️ Error menggunakan shared progress tracking: {str(e)}")
            
        # Fallback ke implementasi sederhana
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = value
            
        if message:
            for label_key in ['overall_label', 'step_label', 'progress_message']:
                if label_key in ui_components:
                    ui_components[label_key].value = message
        
    # Update progress tracker jika tersedia
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        if hasattr(tracker, 'update'):
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
    for button_key in ['download_button', 'check_button', 'reset_button']:
        if button_key in ui_components and hasattr(ui_components[button_key], 'layout'):
            ui_components[button_key].layout.display = 'block'
    
    # Reset progress bar
    _reset_progress_bar(ui_components)
    
    # Update status panel jika tersedia
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], 'Download selesai', 'success')
    elif 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](ui_components, 'info', 'Download selesai')
    
    # Cleanup UI jika tersedia
    if 'cleanup_ui' in ui_components and callable(ui_components['cleanup_ui']):
        ui_components['cleanup_ui'](ui_components)
    elif 'cleanup' in ui_components and callable(ui_components['cleanup']):
        ui_components['cleanup']()