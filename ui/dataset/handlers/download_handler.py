"""
File: smartcash/ui/dataset/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dari berbagai sumber
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display

def handle_download_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dataset.
    
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
        
        # Tampilkan progress bar
        _show_progress(ui_components, "Memulai download dataset...")
        
        # Download dataset dengan thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_download_dataset, ui_components, endpoint_config)
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
    
    except Exception as e:
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat download dataset: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")
    
    finally:
        # Aktifkan kembali tombol
        _disable_ui_buttons(ui_components, False)

def _download_dataset(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Proses download dataset berdasarkan konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download
        
    Returns:
        Tuple (success, message)
    """
    endpoint_type = config.get('type')
    
    # Update progress
    _update_progress(ui_components, 10, f"Mempersiapkan download dari {endpoint_type}...")
    
    # Jalankan downloader berdasarkan tipe endpoint
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
    tracker_key = 'dataset_downloader_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)

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