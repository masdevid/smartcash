"""
File: smartcash/ui/dataset/download/handlers/cleanup_handler.py
Deskripsi: Handler untuk proses cleanup dataset
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, clear_output
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.dataset.manager import DatasetManager
from smartcash.dataset.services.downloader.download_service import DownloadService

def cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """Membersihkan UI setelah proses selesai."""
    # Gunakan reset_progress dari shared component jika tersedia
    try:
        from smartcash.ui.components.progress_tracking import reset_progress
        reset_progress(ui_components)
    except Exception as e:
        # Fallback ke implementasi sederhana
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"⚠️ Error menggunakan shared progress reset: {str(e)}")
            
        # Reset progress bar
        if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
            ui_components['progress_bar'].value = 0
            if hasattr(ui_components['progress_bar'], 'description'):
                ui_components['progress_bar'].description = "Progress: 0%"
            if hasattr(ui_components['progress_bar'], 'layout'):
                ui_components['progress_bar'].layout.visibility = 'hidden'
        
        # Reset label progress
        for label_key in ['step_label', 'overall_label', 'progress_message']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
                ui_components[label_key].value = ""
                if hasattr(ui_components[label_key], 'layout'):
                    ui_components[label_key].layout.visibility = 'hidden'
    
    # Aktifkan kembali tombol yang dinonaktifkan
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button']
    for btn_key in button_keys:
        if btn_key in ui_components and hasattr(ui_components[btn_key], 'disabled'):
            ui_components[btn_key].disabled = False
            if hasattr(ui_components[btn_key], 'layout'):
                ui_components[btn_key].layout.display = 'inline-block'
    
    # Sembunyikan tombol stop jika ada
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'none'
    
    # Set flag running ke False
    ui_components['download_running'] = False

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update panel status tanpa menghapus output sebelumnya.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status (success, error, info, warning)
        message: Pesan status
    """
    # Gunakan status_panel jika tersedia
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.utils.alert_utils import update_status_panel as update_panel
            update_panel(ui_components['status_panel'], message, status_type)
            return
        except Exception as e:
            # Log error jika ada logger
            logger = ui_components.get('logger')
            if logger:
                logger.debug(f"⚠️ Error menggunakan shared status panel: {str(e)}")
    
    # Fallback ke implementasi sederhana
    if 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator(status_type, message))

def setup_download_cleanup_handler(ui_components: Dict[str, Any], module_type: str = 'download', config: Optional[Dict[str, Any]] = None, env: Any = None) -> Dict[str, Any]:
    """
    Setup handler untuk membersihkan UI setelah proses download selesai.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul (default: 'download')
        config: Konfigurasi modul (opsional)
        env: Environment (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger
    logger = ui_components.get('logger', get_logger('download_cleanup'))
    
    # Coba gunakan shared processing_cleanup_handler jika tersedia
    try:
        from smartcash.ui.handlers.processing_cleanup_handler import setup_processing_cleanup_handler
        ui_components = setup_processing_cleanup_handler(ui_components, module_type)
        logger.debug(f"✅ Menggunakan shared processing cleanup handler untuk {module_type}")
    except ImportError as e:
        logger.debug(f"ℹ️ Shared processing cleanup handler tidak tersedia: {str(e)}")
        # Fallback ke implementasi lokal
        ui_components['cleanup_ui'] = cleanup_ui
        ui_components['update_status_panel'] = update_status_panel
    
    # Tambahkan handler untuk tombol cleanup jika tersedia
    if 'cleanup_button' in ui_components:
        try:
            from smartcash.ui.dataset.download.handlers.cleanup_button_handler import handle_cleanup_button_click
            ui_components['cleanup_button'].on_click(lambda b: handle_cleanup_button_click(b, ui_components))
            logger.debug("✅ Handler tombol cleanup berhasil ditambahkan")
        except ImportError as e:
            logger.debug(f"ℹ️ Cleanup button handler tidak tersedia: {str(e)}")
            # Sembunyikan tombol cleanup jika handler tidak tersedia
            if hasattr(ui_components['cleanup_button'], 'layout'):
                ui_components['cleanup_button'].layout.display = 'none'
    
    # Set flag running ke False
    ui_components['download_running'] = False
    
    return ui_components

class CleanupHandler:
    """Handler untuk proses cleanup dataset."""
    def __init__(self, ui_components: Dict[str, Any], dataset_manager=None):
        self.ui_components = ui_components
        self.dataset_manager = dataset_manager or DatasetManager()
    
    def cleanup(self) -> None:
        output_dir = self.ui_components.get('output_dir', {}).value if 'output_dir' in self.ui_components else 'data'
        self.dataset_manager.cleanup_dataset(output_dir=output_dir)
