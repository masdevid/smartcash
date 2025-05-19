"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Inisialisasi UI untuk download dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.components.observer import get_observer_manager
from smartcash.ui.dataset.download.components import create_download_ui
from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.download.utils.notification_manager import (
    notify_log,
    notify_progress,
    DownloadUIEvents
)
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.handlers.config_handler import (
    load_config,
    save_config,
    update_config_from_ui,
    update_ui_from_config
)

def initialize_dataset_download_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk download dataset
    
    Args:
        config: Konfigurasi opsional untuk UI
        
    Returns:
        Dictionary berisi komponen UI
    """
    # Setup logger
    logger = get_logger(__name__)
    
    # Dapatkan observer manager
    observer_manager = get_observer_manager()
    
    # Buat UI components
    ui_components = create_download_ui(config)
    
    # Daftarkan observer
    register_ui_observers(ui_components, observer_manager)
    
    # Tambahkan observer manager ke UI components
    ui_components['observer_manager'] = observer_manager
    
    # Tambahkan fungsi notifikasi ke UI components
    ui_components['notify_log'] = notify_log
    ui_components['notify_progress'] = notify_progress
    
    # Inisialisasi download service
    download_service = DownloadService(
        output_dir=ui_components.get('output_dir', {}).get('value', 'data'),
        config=config,
        logger=logger
    )
    
    # Tambahkan download service ke UI components
    ui_components['download_service'] = download_service
    
    # Setup handlers untuk UI events
    ui_components = setup_download_handlers(ui_components, config=config)
    
    # Load konfigurasi dan update UI
    try:
        loaded_config = load_config()
        update_ui_from_config(loaded_config, ui_components)
    except Exception as e:
        logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
    
    # Setup cleanup function
    def cleanup_resources():
        """Fungsi untuk membersihkan resources."""
        try:
            # Hapus observer
            if observer_manager:
                observer_manager.clear_observers()
            
            # Reset UI components
            if 'progress_container' in ui_components:
                ui_components['progress_container'].layout.display = 'none'
            
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = "Siap untuk download dataset"
            
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
                ui_components['log_output'].clear_output()
                
            logger.info("🧹 Resources berhasil dibersihkan")
        except Exception as e:
            logger.error(f"❌ Error saat membersihkan resources: {str(e)}")
    
    # Tambahkan cleanup function ke UI components
    ui_components['cleanup'] = cleanup_resources
    
    return ui_components