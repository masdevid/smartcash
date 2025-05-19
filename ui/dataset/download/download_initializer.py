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
    load_config,
    save_config,
    update_config_from_ui,
    update_ui_from_config
)
from pathlib import Path
from smartcash.common.config import ConfigManager
from smartcash.common.environment import EnvironmentManager

def get_config_manager() -> ConfigManager:
    """Get the config manager instance."""
    return ConfigManager()

def get_environment_manager() -> EnvironmentManager:
    """Get the environment manager instance."""
    return EnvironmentManager()

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
            loaded_config = load_config()
            update_ui_from_config(loaded_config, ui_components)
            
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
        
        # Setup cleanup function
        def cleanup_resources():
            """Fungsi untuk membersihkan resources."""
            try:
                # Notifikasi cleanup
                notify_service_event(
                    "download",
                    "progress",
                    ui_components,
                    observer_manager,
                    message="Membersihkan resources...",
                    step="cleanup"
                )
                # Hapus observer
                if observer_manager:
                    # Use unregister_all if available
                    if hasattr(observer_manager, 'unregister_all'):
                        observer_manager.unregister_all()
                    elif hasattr(observer_manager, 'clear_observers'):
                        observer_manager.clear_observers()
                # Reset UI components
                if 'progress_container' in ui_components:
                    ui_components['progress_container'].layout.display = 'none'
                if 'status_panel' in ui_components:
                    ui_components['status_panel'].value = "Siap untuk download dataset"
                if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
                    ui_components['log_output'].clear_output()
                # Notifikasi cleanup selesai
                notify_service_event(
                    "download",
                    "complete",
                    ui_components,
                    observer_manager,
                    message="Resources berhasil dibersihkan",
                    step="cleanup"
                )
                logger.info("🧹 Resources berhasil dibersihkan")
            except Exception as e:
                error_msg = f"Error saat membersihkan resources: {str(e)}"
                # Use logger from ui_components if available
                if 'logger' in ui_components and hasattr(ui_components['logger'], 'error'):
                    ui_components['logger'].error(error_msg)
                logger.error(f"❌ {error_msg}")
                # Notifikasi error cleanup
                notify_service_event(
                    "download",
                    "error",
                    ui_components,
                    observer_manager,
                    message=error_msg,
                    step="cleanup"
                )
        
        # Tambahkan cleanup function ke UI components
        ui_components['cleanup'] = cleanup_resources
        
        # Pastikan semua komponen terhubung dengan benar
        main_ui = ui_components['ui']
        if not isinstance(main_ui, widgets.VBox):
            raise ValueError("UI component must be a VBox widget")
            
        # Set layout untuk memastikan tampilan yang benar
        main_ui.layout = widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        )
        
        # Redirect logger output ke log widget
        class LogRedirector:
            def __init__(self, log_output):
                self.log_output = log_output
                self.buffer = []
            
            def write(self, message):
                if message.strip():
                    self.buffer.append(message)
                    with self.log_output:
                        from IPython.display import display
                        display(widgets.HTML(f"<p>{message}</p>"))
            
            def flush(self):
                pass
        
        # Redirect logger ke log widget
        import sys
        log_redirector = LogRedirector(ui_components['log_output'])
        sys.stdout = log_redirector
        sys.stderr = log_redirector
        
        return main_ui
        
    except Exception as e:
        # Log error and re-raise
        logger.error(f"Error saat inisialisasi UI: {str(e)}")
        # Notifikasi error inisialisasi
        notify_service_event(
            "download",
            "error",
            {},  # UI components may not be available
            None,  # Observer manager may not be available
            message=f"Gagal memuat konfigurasi: {str(e)}",
            step="config"
        )
        raise