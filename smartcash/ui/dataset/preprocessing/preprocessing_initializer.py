"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer untuk modul preprocessing dataset dengan pendekatan DRY
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.utils.constants import ICONS

# Konstanta untuk namespace logger
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    PREPROCESSING_LOGGER_NAMESPACE,
    MODULE_LOGGER_NAME,
    get_observer_manager
)

# Import handlers
from smartcash.ui.dataset.preprocessing.handlers.setup_handlers import setup_preprocessing_handlers

# Import utils yang dibutuhkan
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.preprocessing.utils.progress_manager import setup_multi_progress, reset_progress_bar
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel

# Import komponen UI
from smartcash.ui.dataset.preprocessing.components.preprocessing_component import create_preprocessing_ui

# Flag global untuk mencegah inisialisasi ulang
_PREPROCESSING_MODULE_INITIALIZED = False
_INITIALIZED_COMPONENTS = {}

def initialize_preprocessing_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk preprocessing dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    global _PREPROCESSING_MODULE_INITIALIZED, _INITIALIZED_COMPONENTS
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu dengan memeriksa singleton
    if _PREPROCESSING_MODULE_INITIALIZED and _INITIALIZED_COMPONENTS:
        logger.debug(f"{ICONS['info']} UI preprocessing dataset sudah diinisialisasi sebelumnya, menggunakan inisialisasi yang sudah ada")
        return _INITIALIZED_COMPONENTS.get('ui')
    
    # Set flag inisialisasi
    _PREPROCESSING_MODULE_INITIALIZED = True
    logger.info(f"{ICONS['start']} Memulai inisialisasi UI preprocessing dataset")
    
    try:
        # Get config manager dengan fallback otomatis
        config_manager = get_config_manager()
        
        # Get dataset config dari SimpleConfigManager
        dataset_config = config_manager.get_module_config('dataset')
        
        # Merge dengan config yang diberikan
        if config:
            # Update konfigurasi yang ada dengan konfigurasi baru
            if not dataset_config:
                dataset_config = {}
            
            # Hanya update kunci yang ada di level pertama
            for key, value in config.items():
                if isinstance(value, dict) and key in dataset_config and isinstance(dataset_config[key], dict):
                    # Update nested dict tanpa mengganti seluruh dict
                    dataset_config[key].update(value)
                else:
                    # Ganti nilai dengan nilai baru
                    dataset_config[key] = value
            
        # Buat UI components
        ui_components = create_preprocessing_ui(dataset_config)
        
        # Setup logger dengan namespace spesifik
        ui_components = setup_ui_logger(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
        
        # Tambahkan UI flag
        ui_components['preprocessing_initialized'] = True
        ui_components['preprocessing_running'] = False
        
        # Setup observer
        observer_manager = get_observer_manager()
        ui_components['observer_manager'] = observer_manager
        ui_components['observer_group'] = "preprocessing_observers"
        
        # Register UI observer
        register_ui_observers(ui_components)
        
        # Setup handlers dengan metode baru
        ui_components = setup_preprocessing_handlers(ui_components, env, dataset_config)
        
        # Set status panel awal
        update_status_panel(ui_components, "idle", "Preprocessing siap dijalankan")
        
        # Tambahkan ke cache singleton
        _INITIALIZED_COMPONENTS = ui_components
        
        # Log setup berhasil
        log_message(ui_components, "UI preprocessing dataset berhasil diinisialisasi", "success", "✅")
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat inisialisasi UI preprocessing: {str(e)}")
        # Reset flag inisialisasi jika gagal
        _PREPROCESSING_MODULE_INITIALIZED = False
        
        # Buat temporary UI untuk menampilkan error
        try:
            from smartcash.ui.utils.alert_utils import create_error_alert
            import ipywidgets as widgets
            
            error_alert = create_error_alert(
                f"Error saat inisialisasi UI preprocessing: {str(e)}",
                "Initialization Error"
            )
            
            ui = widgets.VBox([
                widgets.HTML(f"<h2>{ICONS['error']} Error Inisialisasi UI Preprocessing</h2>"),
                error_alert
            ])
            
            return ui
        except:
            # Jika gagal membuat UI error, raise exception original
            raise

def get_preprocessing_ui_components() -> Dict[str, Any]:
    """
    Dapatkan UI components yang sudah diinisialisasi sebelumnya.
    
    Returns:
        Dictionary komponen UI atau None jika belum diinisialisasi
    """
    global _INITIALIZED_COMPONENTS
    
    if not _INITIALIZED_COMPONENTS:
        logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
        logger.warning(f"{ICONS['warning']} UI preprocessing dataset belum diinisialisasi")
        return None
    
    return _INITIALIZED_COMPONENTS

def reset_preprocessing_ui(env=None, config=None) -> Any:
    """
    Reset UI preprocessing dataset ke kondisi awal.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang baru
    """
    global _PREPROCESSING_MODULE_INITIALIZED, _INITIALIZED_COMPONENTS
    
    # Reset flag inisialisasi
    _PREPROCESSING_MODULE_INITIALIZED = False
    _INITIALIZED_COMPONENTS = {}
    
    # Inisialisasi ulang
    return initialize_preprocessing_ui(env, config)

def initialize_preprocessing_detail_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI preprocessing dataset dengan detail komponen.
    
    Args:
        config: Konfigurasi UI (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    global _PREPROCESSING_MODULE_INITIALIZED
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _PREPROCESSING_MODULE_INITIALIZED:
        logger.debug("UI preprocessing dataset sudah diinisialisasi sebelumnya")
    else:
        logger.info("Memulai inisialisasi UI preprocessing dataset (detail)")
        _PREPROCESSING_MODULE_INITIALIZED = True
    
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
    ui_components = create_preprocessing_ui(config)
    
    # Setup logger dengan namespace spesifik
    logger = create_ui_logger(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
    ui_components['logger'] = logger
    ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
    ui_components['preprocessing_initialized'] = True
    
    # Tambahkan flag untuk tracking status
    ui_components['preprocessing_running'] = False
    ui_components['cleanup_running'] = False
    
    # Register observer untuk notifikasi
    observer_manager = register_ui_observers(ui_components)
    ui_components['observer_manager'] = observer_manager
    
    # Setup multi-progress tracking
    setup_multi_progress(ui_components)
    
    # Setup handlers dengan metode baru
    ui_components = setup_preprocessing_handlers(ui_components, None, config)
    
    logger.info("UI preprocessing dataset berhasil diinisialisasi")
    
    return ui_components