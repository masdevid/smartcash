"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk UI augmentasi dataset dengan logger bridge yang sudah diperbaiki
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.dataset.augmentation.utils.logger_helper import is_initialized

# Konstanta untuk namespace logger
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"

# Import handlers dengan logger bridge
from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_augmentation_button_click
from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import handle_cleanup_button_click
from smartcash.ui.dataset.augmentation.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.augmentation.handlers.save_handler import handle_save_button_click

# Import utils yang dibutuhkan
from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.augmentation.utils.progress_manager import setup_multi_progress

# Import komponen UI
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui

# Flag global untuk mencegah inisialisasi ulang
_AUGMENTATION_MODULE_INITIALIZED = False

def initialize_dataset_augmentation_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset augmentation dengan logger bridge.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama
    """
    global _AUGMENTATION_MODULE_INITIALIZED
    
    # Setup logger bridge untuk komunikasi UI-Service
    ui_components = {'logger_namespace': AUGMENTATION_LOGGER_NAMESPACE}
    ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi
    if _AUGMENTATION_MODULE_INITIALIZED:
        ui_logger.debug("🔄 UI augmentasi sudah diinisialisasi")
    else:
        ui_logger.info("🚀 Memulai inisialisasi UI augmentasi dataset")
        _AUGMENTATION_MODULE_INITIALIZED = True
    
    try:
        # Get config manager dengan fallback
        config_manager = get_config_manager()
        
        # Load augmentation config
        try:
            augmentation_config = config_manager.get_module_config('augmentation')
        except Exception as e:
            ui_logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
            augmentation_config = {}
        
        # Merge dengan config yang diberikan
        if config:
            augmentation_config.update(config)
            
        # Create UI components
        ui_components = create_augmentation_ui(env, augmentation_config)
        
        # Setup logger bridge dengan UI components yang sudah dibuat
        ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = ui_logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
        
        # Setup handlers dan progress
        ui_components = setup_augmentation_handlers(ui_components, env, augmentation_config)
        
        ui_logger.success("✅ UI augmentasi berhasil diinisialisasi")
        return ui_components['ui']
        
    except Exception as e:
        ui_logger.error(f"🔥 Error inisialisasi UI: {str(e)}")
        raise

def initialize_augmentation_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI augmentasi dengan handlers lengkap dan logger bridge.
    
    Args:
        config: Konfigurasi UI (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    global _AUGMENTATION_MODULE_INITIALIZED
    
    # Setup logger bridge terlebih dahulu
    ui_components = {'logger_namespace': AUGMENTATION_LOGGER_NAMESPACE}
    ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
    
    if _AUGMENTATION_MODULE_INITIALIZED:
        ui_logger.debug("🔄 UI augmentasi sudah diinisialisasi")
    else:
        ui_logger.info("🚀 Memulai inisialisasi UI augmentasi (detail)")
        _AUGMENTATION_MODULE_INITIALIZED = True
    
    try:
        # Load config dari ConfigManager jika tidak diberikan
        if config is None:
            config_manager = get_config_manager()
            try:
                config = config_manager.get_module_config('augmentation')
            except Exception as e:
                ui_logger.warning(f"⚠️ Gagal memuat config: {str(e)}")
                config = {}
        
        # Buat UI components
        ui_components = create_augmentation_ui(None, config)
        
        # Setup logger bridge dengan UI components yang sudah dibuat
        ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = ui_logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
        
        # Tambahkan state tracking flags
        ui_components['augmentation_running'] = False
        ui_components['cleanup_running'] = False
        ui_components['stop_requested'] = False
        
        # Setup observer dengan error handling
        try:
            observer_manager = register_ui_observers(ui_components)
            ui_components['observer_manager'] = observer_manager
        except Exception as e:
            ui_logger.warning(f"⚠️ Gagal setup observer: {str(e)}")
        
        # Setup button handlers
        _setup_button_handlers(ui_components)
        
        ui_logger.success("✅ UI augmentasi berhasil diinisialisasi")
        return ui_components
        
    except Exception as e:
        ui_logger.error(f"🔥 Error inisialisasi: {str(e)}")
        raise

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI augmentasi dengan logger bridge.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Pastikan logger bridge tersedia
    if 'logger' not in ui_components:
        ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = ui_logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
    else:
        ui_logger = ui_components['logger']
    
    # Setup observer untuk notifikasi
    try:
        observer_manager = register_ui_observers(ui_components)
        ui_components['observer_manager'] = observer_manager
    except Exception as e:
        ui_logger.warning(f"⚠️ Observer setup failed: {str(e)}")
    
    # Setup handlers untuk UI events
    _setup_button_handlers(ui_components)
    
    # Setup progress tracking
    try:
        setup_multi_progress(ui_components)
    except Exception as e:
        ui_logger.warning(f"⚠️ Progress setup error: {str(e)}")
    
    # Save config ke UI components
    ui_components['config'] = config or {}
    
    # Load dan update UI dari config jika ada
    try:
        if config and isinstance(config, dict):
            from smartcash.ui.dataset.augmentation.handlers.config_handler import _update_ui_from_config_values
            _update_ui_from_config_values(ui_components, config, ui_logger)
    except Exception as e:
        ui_logger.warning(f"⚠️ Config UI update failed: {str(e)}")
    
    ui_logger.success("✅ Handlers berhasil disetup")
    return ui_components

def _setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol UI dengan logger bridge."""
    
    # Setup augmentation button dengan wrapper yang menggunakan logger bridge
    if 'augment_button' in ui_components:
        ui_components['augment_button'].on_click(
            lambda b: handle_augmentation_button_click(ui_components, b)
        )
    
    # Setup cleanup button
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(
            lambda b: handle_cleanup_button_click(ui_components, b)
        )
    
    # Setup reset button
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button_click(ui_components, b)
        )
    
    # Setup save button
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button_click(ui_components, b)
        )