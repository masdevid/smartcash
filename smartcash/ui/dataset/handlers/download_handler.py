"""
File: smartcash/ui/dataset/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dari berbagai sumber dengan integrasi service downloader
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display

def handle_download_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dataset dengan konfirmasi.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # Validasi input
    from smartcash.ui.dataset.handlers.validator import validate_download_config
    
    # Dapatkan konfigurasi endpoint dari UI
    from smartcash.ui.dataset.handlers.endpoint_handler import get_endpoint_config
    endpoint_config = get_endpoint_config(ui_components)
    
    # Cek API key untuk Roboflow jika diperlukan
    if endpoint_config.get('type') == 'roboflow':
        from smartcash.ui.dataset.handlers.api_key_handler import check_api_key
        has_key, api_key = check_api_key(ui_components)
        
        if not has_key:
            # Tampilkan pesan input API key
            from smartcash.ui.dataset.handlers.api_key_handler import request_api_key_input
            request_api_key_input(ui_components)
            return
        
        # Update API key di konfigurasi
        endpoint_config['api_key'] = api_key
    
    # Validasi konfigurasi
    is_valid, error_msg = validate_download_config(endpoint_config)
    if not is_valid:
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
        return
    
    try:
        # Nonaktifkan tombol selama proses
        _disable_ui_buttons(ui_components, True)
        
        # Buat fungsi untuk download dengan ThreadPoolExecutor
        def execute_download(config):
            # Tampilkan progress bar
            _show_progress(ui_components, "Memulai download dataset...")
            
            # Download dataset dengan service atau handler yang sesuai
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_download_dataset, ui_components, config)
                success, message = future.result()
                
                # Update UI berdasarkan hasil
                status_type = "success" if success else "error"
                icon = "✅" if success else "❌"
                
                from smartcash.ui.utils.ui_logger import log_to_ui
                log_to_ui(ui_components, message, status_type, icon)
                
                if logger:
                    if success:
                        logger.success(f"{icon} {message}")
                    else:
                        logger.error(f"{icon} {message}")
                
                # Aktifkan kembali tombol
                _disable_ui_buttons(ui_components, False)
        
        # Import konfirmasi handler dan tampilkan konfirmasi jika perlu
        from smartcash.ui.dataset.handlers.confirmation_handler import create_dataset_confirmation, prepare_dataset_confirmation
        proceed_callback, cancel_callback = prepare_dataset_confirmation(ui_components, endpoint_config, execute_download)
        
        # Periksa apakah perlu konfirmasi berdasarkan dataset yang ada
        create_dataset_confirmation(ui_components, proceed_callback, cancel_callback)
    
    except Exception as e:
        # Tampilkan error dan aktifkan kembali tombol
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat download dataset: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
        
        _disable_ui_buttons(ui_components, False)

def _download_dataset(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Proses download dataset berdasarkan konfigurasi dengan dukungan services.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download
        
    Returns:
        Tuple (success, message)
    """
    endpoint_type = config.get('type')
    logger = ui_components.get('logger')
    start_time = time.time()
    
    try:
        # Update progress
        _update_progress(ui_components, 10, f"Mempersiapkan download dari {endpoint_type}...")
        
        # Coba menggunakan DownloadService yang sudah ada
        try:
            from smartcash.dataset.services.downloader.download_service import DownloadService
            from smartcash.components.observer import notify, EventTopics
            
            # Buat konfigurasi yang sesuai untuk service
            output_dir = config.get('output_dir', 'data')
            service_config = {'data': {'roboflow': {}}}
            backup_existing = config.get('backup_existing', False)
            
            # Setup service dengan logger untuk integrasi
            download_service = DownloadService(output_dir=output_dir, config=service_config, logger=logger)
            
            # Jalankan download berdasarkan tipe endpoint
            if endpoint_type == 'roboflow':
                # Pakai service untuk Roboflow
                api_key = config.get('api_key')
                workspace = config.get('workspace')
                project = config.get('project')
                version = config.get('version')
                
                # Setup progress tracking callback untuk integrasi
                def progress_callback(progress=None, message=None, **kwargs):
                    if progress is not None:
                        _update_progress(ui_components, progress, message or "Download dataset...")
                
                # Link observer dengan callback lokal jika memungkinkan
                try:
                    observer_manager = ui_components.get('observer_manager')
                    if observer_manager:
                        observer = observer_manager.create_simple_observer(
                            EventTopics.DOWNLOAD_PROGRESS,
                            lambda event_type, sender, **kwargs: progress_callback(**kwargs),
                            name="UI_Download_Progress",
                            group="dataset_download"
                        )
                except Exception as e:
                    if logger: logger.debug(f"ℹ️ Progress observer tidak tersedia: {str(e)}")
                
                # Update progress sebelum download
                _update_progress(ui_components, 30, f"Mendownload dataset dari Roboflow: {workspace}/{project}:{version}")
                
                # Lakukan download
                result = download_service.download_from_roboflow(
                    api_key=api_key,
                    workspace=workspace,
                    project=project,
                    version=version,
                    output_dir=output_dir,
                    show_progress=True,
                    backup_existing=backup_existing
                )
                
                # Pastikan progress mencapai 100%
                _update_progress(ui_components, 100, "Download selesai")
                
                # Format pesan sukses
                total_images = result.get('stats', {}).get('total_images', 0)
                elapsed_time = time.time() - start_time
                success_msg = f"Dataset berhasil didownload: {total_images} gambar ({elapsed_time:.1f}s)"
                return True, success_msg
                
            elif endpoint_type == 'drive':
                from smartcash.ui.dataset.handlers.gdrive_handler import download_from_drive
                return download_from_drive(ui_components, config)
                
            elif endpoint_type == 'url':
                from smartcash.ui.dataset.handlers.url_handler import download_from_url
                return download_from_url(ui_components, config)
                
            else:
                return False, f"Tipe endpoint tidak dikenal: {endpoint_type}"
                
        except ImportError:
            # Fallback jika service tidak tersedia
            if logger: logger.warning(f"⚠️ DownloadService tidak tersedia, menggunakan fallback handlers")
            
            # Fallback ke handler spesifik
            if endpoint_type == 'roboflow':
                from smartcash.ui.dataset.handlers.roboflow_handler import download_from_roboflow
                return download_from_roboflow(ui_components, config)
            elif endpoint_type == 'drive':
                from smartcash.ui.dataset.handlers.gdrive_handler import download_from_drive
                return download_from_drive(ui_components, config)
            elif endpoint_type == 'url':
                from smartcash.ui.dataset.handlers.url_handler import download_from_url
                return download_from_url(ui_components, config)
            else:
                return False, f"Tipe endpoint tidak dikenal: {endpoint_type}"
    
    except Exception as e:
        if logger: logger.error(f"❌ Error saat download dataset: {str(e)}")
        return False, f"Error saat download dataset: {str(e)}"

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
    Update progress bar dan progress tracker.
    
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
    tracker_key = 'dataset_downloader_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Juga coba notify melalui observer jika tersedia
    try:
        from smartcash.components.observer import notify, EventTopics
        notify(EventTopics.DOWNLOAD_PROGRESS, "ui_downloader", 
              progress=value, total=100, message=message, status="info")
    except ImportError:
        pass

def _disable_ui_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
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