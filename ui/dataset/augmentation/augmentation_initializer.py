"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk UI augmentasi dataset yang mengikuti pola dataset download
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger

# Konstanta untuk namespace logger
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"
MODULE_LOGGER_NAME = "AUGMENTATION"

# Import handlers dengan nama yang lebih spesifik
from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_augmentation_button_click
from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import handle_cleanup_button_click
from smartcash.ui.dataset.augmentation.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.augmentation.handlers.save_handler import handle_save_button_click

# Import utils yang dibutuhkan untuk inisialisasi
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, is_initialized
from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel, reset_ui_after_augmentation
from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar

# Import modul untuk setup
from smartcash.ui.dataset.augmentation.handlers.setup_handlers import setup_augmentation_handlers
from smartcash.ui.dataset.augmentation.components import create_augmentation_ui

# Flag global untuk mencegah inisialisasi ulang
_AUGMENTATION_MODULE_INITIALIZED = False

def initialize_dataset_augmentation_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset augmentation.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    global _AUGMENTATION_MODULE_INITIALIZED
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _AUGMENTATION_MODULE_INITIALIZED:
        logger.debug("UI augmentasi dataset sudah diinisialisasi sebelumnya, menggunakan inisialisasi yang sudah ada")
    else:
        logger.info("Memulai inisialisasi UI augmentasi dataset")
        _AUGMENTATION_MODULE_INITIALIZED = True
    
    try:
        # Get config manager dengan fallback otomatis
        config_manager = get_config_manager()
        
        # Get augmentation config dari SimpleConfigManager
        augmentation_config = config_manager.get_module_config('augmentation')
        
        # Merge dengan config yang diberikan
        if config:
            augmentation_config.update(config)
            
        # Create UI components
        ui_components = create_augmentation_ui(augmentation_config)
        
        # Setup logger dengan namespace spesifik
        logger = create_ui_logger(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
        
        # Setup handlers
        ui_components = setup_augmentation_handlers(ui_components, env, augmentation_config)
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"Error saat inisialisasi UI: {str(e)}")
        # Hanya tampilkan log di UI jika bukan saat dependency installer
        try:
            # Buat temporary ui_components untuk log error
            temp_ui_components = {'logger': logger, 'logger_namespace': AUGMENTATION_LOGGER_NAMESPACE}
            log_message(temp_ui_components, f"Error saat inisialisasi UI: {str(e)}", "error")
        except:
            pass
        raise

def initialize_augmentation_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI augmentasi dataset.
    
    Args:
        config: Konfigurasi UI (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    global _AUGMENTATION_MODULE_INITIALIZED
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _AUGMENTATION_MODULE_INITIALIZED:
        logger.debug("UI augmentasi dataset sudah diinisialisasi sebelumnya")
    else:
        logger.info("Memulai inisialisasi UI augmentasi dataset (detail)")
        _AUGMENTATION_MODULE_INITIALIZED = True
    
    try:
        # Get config dari SimpleConfigManager jika config tidak diberikan
        if config is None:
            config_manager = get_config_manager()
            config = config_manager.get_module_config('augmentation')
    except Exception as e:
        logger.warning(f"Gagal memuat konfigurasi dari SimpleConfigManager: {str(e)}")
        # Gunakan config kosong sebagai fallback
        config = {}
    
    # Buat UI components
    ui_components = create_augmentation_ui(config)
    
    # Setup logger dengan namespace spesifik
    logger = create_ui_logger(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
    ui_components['logger'] = logger
    ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
    ui_components['augmentation_initialized'] = True
    
    # Tambahkan flag untuk tracking status
    ui_components['augmentation_running'] = False
    ui_components['cleanup_running'] = False
    
    # Register observer untuk notifikasi
    observer_manager = register_ui_observers(ui_components)
    ui_components['observer_manager'] = observer_manager
    
    # Tambahkan handler untuk tombol
    ui_components['augment_button'].on_click(
        lambda b: handle_augmentation_button_click(ui_components, b)
    )
    
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(
            lambda b: handle_cleanup_button_click(ui_components, b)
        )
    
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button_click(ui_components, b)
        )
    
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button_click(ui_components, b)
        )
    
    logger.info("UI augmentasi dataset berhasil diinisialisasi")
    
    return ui_components