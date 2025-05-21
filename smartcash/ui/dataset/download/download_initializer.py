"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer untuk UI download dataset
"""

from typing import Dict, Any, Optional
# Path tidak digunakan langsung tetapi mungkin dibutuhkan untuk type hints di komponen lain
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger

# Konstanta untuk namespace logger
DOWNLOAD_LOGGER_NAMESPACE = "smartcash.dataset.download"
# Konstanta untuk ID namespace di UI
MODULE_LOGGER_NAME = "DATASET"

# Import handlers dengan nama yang lebih spesifik untuk menghindari konflik
from smartcash.ui.dataset.download.handlers.download_handler import handle_download_button_click
from smartcash.ui.dataset.download.handlers.check_handler import handle_check_button_click
from smartcash.ui.dataset.download.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.download.handlers.cleanup_button_handler import handle_cleanup_button_click

# Import utils yang dibutuhkan untuk inisialisasi
from smartcash.ui.dataset.download.utils.logger_helper import log_message, is_initialized
from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
# Fungsi-fungsi berikut tidak digunakan secara langsung tetapi tersedia untuk digunakan oleh handlers
from smartcash.ui.dataset.download.utils.ui_state_manager import update_status_panel, reset_ui_after_download
from smartcash.ui.dataset.download.utils.progress_manager import reset_progress_bar

# Import modul untuk setup
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.components import create_download_ui

# Flag global untuk mencegah inisialisasi ulang
_DOWNLOAD_MODULE_INITIALIZED = False

def initialize_dataset_download_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset downloader.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    global _DOWNLOAD_MODULE_INITIALIZED
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _DOWNLOAD_MODULE_INITIALIZED:
        logger.debug("UI download dataset sudah diinisialisasi sebelumnya, menggunakan inisialisasi yang sudah ada")
    else:
        logger.info("Memulai inisialisasi UI download dataset")
        _DOWNLOAD_MODULE_INITIALIZED = True
    
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
        
        # Setup logger dengan namespace spesifik
        logger = create_ui_logger(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
        ui_components['logger'] = logger
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        # Setup handlers
        ui_components = setup_download_handlers(ui_components, env, dataset_config)
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"Error saat inisialisasi UI: {str(e)}")
        # Hanya tampilkan log di UI jika bukan saat dependency installer
        try:
            # Buat temporary ui_components untuk log error
            temp_ui_components = {'logger': logger, 'logger_namespace': DOWNLOAD_LOGGER_NAMESPACE}
            log_message(temp_ui_components, f"Error saat inisialisasi UI: {str(e)}", "error")
        except:
            pass
        raise

def initialize_download_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI download dataset.
    
    Args:
        config: Konfigurasi UI (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    global _DOWNLOAD_MODULE_INITIALIZED
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _DOWNLOAD_MODULE_INITIALIZED:
        logger.debug("UI download dataset sudah diinisialisasi sebelumnya")
    else:
        logger.info("Memulai inisialisasi UI download dataset (detail)")
        _DOWNLOAD_MODULE_INITIALIZED = True
    
    try:
        # Get config dari SimpleConfigManager jika config tidak diberikan
        if config is None:
            config_manager = get_config_manager()
            config = config_manager.get_module_config('dataset')
    except Exception as e:
        logger.warning(f"Gagal memuat konfigurasi dari SimpleConfigManager: {str(e)}")
        # Gunakan config kosong sebagai fallback
        config = {}
    
    # Buat UI components
    ui_components = create_download_ui(config)
    
    # Setup logger dengan namespace spesifik
    logger = create_ui_logger(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
    ui_components['logger'] = logger
    ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
    ui_components['download_initialized'] = True
    
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
        lambda b: handle_cleanup_button_click(b, ui_components)
    )
    
    logger.info("UI download dataset berhasil diinisialisasi")
    
    return ui_components