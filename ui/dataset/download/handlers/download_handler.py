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
        # Tangani error
        error_msg = f"Error saat download dataset: {str(e)}"
        if download_logger:
            download_logger.error(f"❌ {error_msg}")
            
        # Update UI dengan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)

def _download_from_roboflow(ui_components: Dict[str, Any]) -> None:
    """
    Download dataset dari Roboflow menggunakan dataset manager.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('download_logger') or ui_components.get('logger')
    
    try:
        # Dapatkan konfigurasi endpoint dari UI
        from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
        config = get_endpoint_config(ui_components)
        
        # Log konfigurasi
        if logger:
            logger.info(f"💾 Memulai download dataset dari Roboflow dengan konfigurasi: {config}")
        
        # Update progress
        _update_progress(ui_components, 10, "Menginisialisasi download dari Roboflow...")
        
        # Coba import dataset manager
        try:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager()
        except ImportError as e:
            # Fallback ke implementasi sederhana
            if logger: logger.warning(f"⚠️ DatasetManager tidak tersedia: {str(e)}")
            
            # Coba import roboflow service langsung
            try:
                from smartcash.dataset.services.roboflow_service import RoboflowService
                service = RoboflowService()
                
                # Update progress
                _update_progress(ui_components, 20, "Mendownload dataset dari Roboflow...")
                
                # Download dataset
                result = service.download_dataset(
                    workspace=config.get('workspace'),
                    project=config.get('project'),
                    version=config.get('version'),
                    api_key=config.get('api_key'),
                    format=config.get('format', 'yolov5pytorch'),
                    output_dir=config.get('output_dir')
                )
                
                # Proses hasil download
                _process_download_result(ui_components, result)
                return
            except ImportError as e2:
                # Tidak ada service tersedia
                error_msg = f"Tidak dapat mengimport RoboflowService: {str(e2)}"
                if logger: logger.error(f"❌ {error_msg}")
                
                # Update UI dengan error
                from smartcash.ui.utils.ui_logger import log_to_ui
                log_to_ui(ui_components, error_msg, "error", "❌")
                
                # Reset UI
                _reset_ui_after_download(ui_components)
                return
        
        # Update progress
        _update_progress(ui_components, 20, "Mendownload dataset dari Roboflow...")
        
        # Download dataset menggunakan dataset manager
        result = dataset_manager.download_dataset(
            source="roboflow",
            config=config
        )
        
        # Proses hasil download
        _process_download_result(ui_components, result)
        
    except Exception as e:
        # Tangani error
        error_msg = f"Error saat download dataset: {str(e)}"
        if logger: logger.error(f"❌ {error_msg}")
        
        # Update UI dengan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)

def _process_download_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Proses hasil download dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil download dari dataset service
    """
    logger = ui_components.get('download_logger') or ui_components.get('logger')
    
    # Cek apakah download berhasil
    success = result.get('success', False)
    message = result.get('message', '')
    dataset_info = result.get('dataset_info', {})
    
    if success:
        # Update progress
        _update_progress(ui_components, 100, "Download selesai!")
        
        # Log sukses
        if logger: logger.info(f"✅ {message}")
        
        # Update UI dengan sukses
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, message, "success", "✅")
        
        # Tampilkan informasi dataset
        if dataset_info:
            # Tampilkan informasi dataset di summary container
            if 'summary_container' in ui_components:
                from IPython.display import display, HTML
                import pandas as pd
                
                with ui_components['summary_container']:
                    # Clear output terlebih dahulu
                    ui_components['summary_container'].clear_output()
                    
                    # Tampilkan header
                    display(HTML(f"<h3>Informasi Dataset</h3>"))
                    
                    # Buat DataFrame dari dataset_info
                    df = pd.DataFrame([
                        ["Nama", dataset_info.get('name', 'N/A')],
                        ["Versi", dataset_info.get('version', 'N/A')],
                        ["Format", dataset_info.get('format', 'N/A')],
                        ["Jumlah Gambar", dataset_info.get('total_images', 0)],
                        ["Jumlah Kelas", len(dataset_info.get('classes', []))],
                        ["Kelas", ", ".join(dataset_info.get('classes', []))],
                        ["Path", dataset_info.get('path', 'N/A')]
                    ], columns=["Properti", "Nilai"])
                    
                    # Tampilkan DataFrame
                    display(df)
                
                # Tampilkan summary container
                ui_components['summary_container'].layout.display = 'block'
    else:
        # Update progress dengan error
        _update_progress(ui_components, 0, f"Error: {message}")
        
        # Log error
        if logger: logger.error(f"❌ {message}")
        
        # Update UI dengan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, message, "error", "❌")
    
    # Reset UI setelah proses selesai
    _reset_ui_after_download(ui_components)

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke nilai awal dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Gunakan reset_progress dari shared component jika tersedia
    try:
        from smartcash.ui.components.progress_tracking import reset_progress
        
        reset_progress(ui_components)
    except Exception as e:
        # Log error jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"⚠️ Error menggunakan shared progress tracking: {str(e)}")
            
        # Fallback ke implementasi sederhana
        if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].layout.visibility = 'hidden'
            # Pastikan margin tetap konsisten
            ui_components['progress_bar'].layout.margin = '15px 0'
            
        for label_key in ['overall_label', 'step_label', 'progress_message']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].value = ""
                ui_components[label_key].layout.visibility = 'hidden'
                # Pastikan margin tetap konsisten
                ui_components[label_key].layout.margin = '5px 0'
    
    # Reset tracker jika tersedia
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        if hasattr(tracker, 'reset'):
            tracker.reset()
        elif hasattr(tracker, 'close'):
            tracker.close()
            
    # Reset step tracker jika tersedia
    step_tracker_key = 'download_step_tracker'
    if step_tracker_key in ui_components:
        tracker = ui_components[step_tracker_key]
        if hasattr(tracker, 'reset'):
            tracker.reset()
        elif hasattr(tracker, 'close'):
            tracker.close()
            
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()

def _show_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """
    Tampilkan dan reset progress bar dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan progress awal
    """
    # Set flag download_running ke True
    ui_components['download_running'] = True
    
    # Gunakan show_progress dari shared component jika tersedia
    try:
        from smartcash.ui.components.progress_tracking import show_progress
        
        show_progress(
            ui_components=ui_components,
            message=message,
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
            ui_components['progress_bar'].layout.margin = '15px 0'
            
        for label_key in ['overall_label', 'step_label', 'progress_message']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].value = message
                ui_components[label_key].layout.visibility = 'visible'
                ui_components[label_key].layout.margin = '5px 0'

def _update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """
    Update progress bar dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress opsional
    """
    # Pastikan progress bar terlihat
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
        ui_components['progress_bar'].layout.visibility = 'visible'
    
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
                if label_key in ui_components and hasattr(ui_components[label_key], 'layout'):
                    ui_components[label_key].value = message
                    ui_components[label_key].layout.visibility = 'visible'
        
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
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button']
    
    # Set status disabled untuk semua tombol
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disabled
            
            # Atur visibilitas tombol jika disabled
            if hasattr(ui_components[key], 'layout'):
                if disabled:
                    # Sembunyikan tombol reset dan cleanup saat proses berjalan
                    if key in ['reset_button', 'cleanup_button']:
                        ui_components[key].layout.display = 'none'
                else:
                    # Tampilkan kembali semua tombol dengan konsisten
                    ui_components[key].layout.display = 'inline-block'

def _reset_ui_after_download(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses download selesai."""
    # Aktifkan kembali tombol (fungsi ini juga akan mengatur display='inline-block')
    _disable_buttons(ui_components, False)
    
    # Reset progress bar
    _reset_progress_bar(ui_components)
    
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
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
        
    # Set flag download_running ke False jika ada
    ui_components['download_running'] = False