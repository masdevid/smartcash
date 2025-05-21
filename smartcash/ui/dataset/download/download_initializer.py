"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer untuk UI download dataset
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.components import create_download_ui
from smartcash.ui.utils.ui_logger import log_to_ui
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.download.handlers.download_handler import handle_download_button_click
from smartcash.ui.dataset.download.handlers.check_handler import handle_check_button_click
from smartcash.ui.dataset.download.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.download.handlers.cleanup_handler import handle_cleanup_button_click
from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.download.utils.logger_helper import setup_ui_logger

def initialize_dataset_download_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset downloader.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    try:
        # Get config manager dengan fallback otomatis
        config_manager = get_config_manager()
        
        # Get dataset config dari SimpleConfigManager
        dataset_config = config_manager.get_module_config('dataset')
        
        # Merge dengan config yang diberikan
        if config:
            dataset_config.update(config)
            
        # Create UI components
        ui_components = create_download_ui(dataset_config)
        
        # Setup logger
        ui_components = setup_ui_logger(ui_components)
        
        # Setup handlers
        ui_components = setup_download_handlers(ui_components, env, dataset_config)
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        log_to_ui(None, f"❌ Error saat inisialisasi UI: {str(e)}", "error")
        raise

def initialize_download_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI download dataset.
    
    Args:
        config: Konfigurasi UI (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    # Setup logger
    logger = get_logger()
    
    try:
        # Get config dari SimpleConfigManager jika config tidak diberikan
        if config is None:
            config_manager = get_config_manager()
            config = config_manager.get_module_config('dataset')
    except Exception as e:
        logger.warning(f"⚠️ Gagal memuat konfigurasi dari SimpleConfigManager: {str(e)}")
        # Gunakan config kosong sebagai fallback
        config = {}
    
    # Buat UI components
    ui_components = create_download_ui(config)
    
    # Tambahkan logger ke UI components dan setup UI logger
    ui_components['logger'] = logger
    ui_components = setup_ui_logger(ui_components)
    
    # Tambahkan flag untuk tracking status
    ui_components['download_running'] = False
    ui_components['cleanup_running'] = False
    
    # Register observer untuk notifikasi
    observer_manager = register_ui_observers(ui_components)
    ui_components['observer_manager'] = observer_manager
    
    # Tambahkan handler untuk tombol
    ui_components['download_button'].on_click(
        lambda b: handle_download_button_click(ui_components, b)
    )
    
    ui_components['check_button'].on_click(
        lambda b: handle_check_button_click(ui_components, b)
    )
    
    ui_components['reset_button'].on_click(
        lambda b: handle_reset_button_click(ui_components, b)
    )
    
    ui_components['cleanup_button'].on_click(
        lambda b: handle_cleanup_button_click(ui_components, b)
    )
    
    logger.info("✅ UI download dataset berhasil diinisialisasi")
    
    return ui_components