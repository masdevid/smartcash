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
from smartcash.dataset.services.downloader.notification_utils import notify_service_event
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.handlers.config_handler import (
    get_download_config,
    update_config_from_ui,
    update_ui_from_config
)
from pathlib import Path
from smartcash.common.config import get_config_manager
import os

def initialize_dataset_download_ui(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Inisialisasi UI untuk download dataset
    
    Args:
        config: Konfigurasi opsional untuk UI
        
    Returns:
        Widget VBox berisi UI lengkap
    """
    # Setup logger
    logger = get_logger(__name__)
    
    try:
        # Ensure config is a dictionary
        config = config or {}
        logger.info(f"Memulai inisialisasi UI dengan config: {config}")
        
        # Setup base directory for configuration
        base_dir = config.get('base_dir')
        if not base_dir:
            # Fallback ke root project (5 tingkat di atas file ini)
            base_dir = str(Path(__file__).resolve().parents[5])
            logger.warning(f"base_dir tidak diberikan, fallback ke root project: {base_dir}")
            config['base_dir'] = base_dir
        
        config_file = config.get('config_file')
        if not config_file:
            config_file = str(Path(base_dir) / 'smartcash' / 'configs' / 'dataset_config.yaml')
            logger.warning(f"config_file tidak diberikan, fallback ke: {config_file}")
            config['config_file'] = config_file
        
        # Get config manager and load config
        config_manager = get_config_manager(base_dir=base_dir, config_file=config_file)
        config_manager.load_config()
        
        # Dapatkan observer manager
        observer_manager = get_observer_manager()
        
        # Buat UI components
        ui_components = create_download_ui(config)
        
        # Validasi UI components
        if not isinstance(ui_components, dict):
            raise ValueError("UI components harus berupa dictionary")
            
        if 'main_container' not in ui_components:
            raise ValueError("UI components harus mengandung 'main_container'")
        
        # Daftarkan observer
        register_ui_observers(ui_components, observer_manager)
        
        # Tambahkan observer manager ke UI components
        ui_components['observer_manager'] = observer_manager
        
        # Tambahkan fungsi notifikasi ke UI components
        ui_components['notify_log'] = notify_log
        ui_components['notify_progress'] = notify_progress
        ui_components['notify_service_event'] = notify_service_event
        
        # Inisialisasi download service
        output_dir = (
            ui_components['output_dir'].value 
            if hasattr(ui_components['output_dir'], 'value') 
            else str(ui_components['output_dir'])
        ) if 'output_dir' in ui_components else 'data'
        
        download_service = DownloadService(
            output_dir=output_dir,
            config=config,
            logger=logger
        )
        
        # Tambahkan download service ke UI components
        ui_components['download_service'] = download_service
        
        # Setup handlers untuk UI events
        ui_components = setup_download_handlers(ui_components, config=config)
        
        # Load konfigurasi dan update UI
        try:
            # Get download config
            download_config = get_download_config(ui_components)
            
            # Update UI from config
            update_ui_from_config(ui_components, download_config)
            
            # Notifikasi bahwa konfigurasi berhasil dimuat
            notify_service_event(
                "download",
                "start",
                ui_components,
                observer_manager,
                message="Konfigurasi berhasil dimuat",
                step="config"
            )
        except Exception as e:
            logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
            
            # Notifikasi error konfigurasi
            notify_service_event(
                "download",
                "error",
                ui_components,
                observer_manager,
                message=f"Gagal memuat konfigurasi: {str(e)}",
                step="config"
            )
        
        # Return UI components
        return ui_components['main_container']
        
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        raise