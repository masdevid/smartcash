"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk UI augmentasi dataset dengan logger bridge
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

# Konstanta untuk namespace logger
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE

# Import handlers
from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_augmentation_button_click
from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import handle_cleanup_button_click
from smartcash.ui.dataset.augmentation.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.augmentation.handlers.save_handler import handle_save_button_click

# Import utils
from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar, setup_multi_progress

# Import komponen UI
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui

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
    
    # Setup logger dengan bridge pattern
    logger = get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _AUGMENTATION_MODULE_INITIALIZED:
        logger.debug("UI augmentasi dataset sudah diinisialisasi sebelumnya")
    else:
        logger.info("Memulai inisialisasi UI augmentasi dataset")
        _AUGMENTATION_MODULE_INITIALIZED = True
    
    try:
        # Get config manager dengan fallback otomatis
        config_manager = get_config_manager()
        
        # Get augmentation config dari SimpleConfigManager
        try:
            augmentation_config = config_manager.get_module_config('augmentation')
        except Exception as e:
            logger.warning(f"Gagal memuat konfigurasi augmentasi: {str(e)}")
            augmentation_config = {}
        
        # Merge dengan config yang diberikan
        if config:
            if isinstance(augmentation_config, dict):
                augmentation_config.update(config)
            else:
                augmentation_config = config
            
        # Create UI components
        ui_components = create_augmentation_ui(env, augmentation_config)
        
        # Setup logger dengan bridge pattern untuk mencegah circular dependency
        ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = ui_logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
        
        # Setup handlers
        ui_components = setup_augmentation_handlers(ui_components, env, augmentation_config)
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"Error saat inisialisasi UI: {str(e)}")
        raise

def initialize_augmentation_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI augmentasi dataset dengan handlers lengkap.
    
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
            try:
                config = config_manager.get_module_config('augmentation')
            except Exception as e:
                logger.warning(f"Gagal memuat konfigurasi dari SimpleConfigManager: {str(e)}")
                config = {}
    except Exception as e:
        logger.warning(f"Error config manager: {str(e)}")
        config = {}
    
    # Buat UI components
    ui_components = create_augmentation_ui(None, config)
    
    # Setup logger dengan bridge pattern
    ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
    ui_components['logger'] = ui_logger
    ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
    ui_components['augmentation_initialized'] = True
    
    # Tambahkan flag untuk tracking status
    ui_components['augmentation_running'] = False
    ui_components['cleanup_running'] = False
    ui_components['stop_requested'] = False
    
    # Register observer untuk notifikasi
    try:
        observer_manager = register_ui_observers(ui_components)
        ui_components['observer_manager'] = observer_manager
    except Exception as e:
        ui_logger.warning(f"Gagal setup observer: {str(e)}")
    
    # Setup handlers untuk tombol
    _setup_button_handlers(ui_components)
    
    ui_logger.info("UI augmentasi dataset berhasil diinisialisasi")
    
    return ui_components

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger jika belum
    if 'logger' not in ui_components:
        ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = ui_logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
    
    # Setup observer untuk notifikasi
    try:
        observer_manager = register_ui_observers(ui_components)
        ui_components['observer_manager'] = observer_manager
    except Exception as e:
        ui_components['logger'].warning(f"Gagal setup observer: {str(e)}")
    
    # Setup handlers untuk UI events
    _setup_button_handlers(ui_components)
    
    # Setup progress tracking
    try:
        setup_multi_progress(ui_components)
    except Exception as e:
        ui_components['logger'].warning(f"Error setup progress: {str(e)}")
    
    # Save config ke UI components
    ui_components['config'] = config or {}
    
    # Load konfigurasi dan update UI jika ada
    try:
        if config and isinstance(config, dict):
            from smartcash.ui.dataset.augmentation.handlers.config_handler import update_ui_from_config
            update_ui_from_config(ui_components, config)
    except Exception as e:
        ui_components['logger'].warning(f"Gagal update UI dari config: {str(e)}")
    
    ui_components['logger'].success("✅ Augmentasi dataset handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol UI dengan safe checking."""
    
    logger = ui_components.get('logger')
    
    # Debug: Log available button keys
    if logger:
        available_buttons = [k for k in ui_components.keys() if 'button' in k and ui_components[k] is not None]
        logger.debug(f"Available buttons: {available_buttons}")
        
        # Log action_buttons structure for debugging
        if 'action_buttons' in ui_components:
            action_buttons = ui_components['action_buttons']
            if action_buttons:
                logger.debug(f"Action buttons keys: {list(action_buttons.keys()) if hasattr(action_buttons, 'keys') else 'Not a dict'}")
    
    # Setup augmentation button - Safe checking
    augment_button = ui_components.get('augment_button')
    if augment_button is not None and hasattr(augment_button, 'on_click'):
        augment_button.on_click(
            lambda b: handle_augmentation_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Augment button handler registered")
    else:
        if logger:
            logger.warning("⚠️ Augment button not found or is None - checking action_buttons structure")
            # Try to find the button in action_buttons
            action_buttons = ui_components.get('action_buttons', {})
            if action_buttons:
                # Try common key names
                for key in ['primary_button', 'main_button', 'action_button', 'augment_button']:
                    button = action_buttons.get(key)
                    if button is not None and hasattr(button, 'on_click'):
                        button.on_click(lambda b: handle_augmentation_button_click(ui_components, b))
                        ui_components['augment_button'] = button  # Update reference
                        logger.success(f"✅ Found augment button as '{key}' and registered handler")
                        break
    
    # Setup cleanup button - Safe checking
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button is not None and hasattr(cleanup_button, 'on_click'):
        cleanup_button.on_click(
            lambda b: handle_cleanup_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Cleanup button handler registered")
    else:
        if logger:
            logger.warning("⚠️ Cleanup button not found or is None")
    
    # Setup reset button - Safe checking
    reset_button = ui_components.get('reset_button')
    if reset_button is not None and hasattr(reset_button, 'on_click'):
        reset_button.on_click(
            lambda b: handle_reset_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Reset button handler registered")
    else:
        if logger:
            logger.warning("⚠️ Reset button not found or is None")
    
    # Setup save button - Safe checking
    save_button = ui_components.get('save_button')
    if save_button is not None and hasattr(save_button, 'on_click'):
        save_button.on_click(
            lambda b: handle_save_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Save button handler registered")
    else:
        if logger:
            logger.warning("⚠️ Save button not found or is None")
    
    # Setup stop button if available - Safe checking
    stop_button = ui_components.get('stop_button')
    if stop_button is not None and hasattr(stop_button, 'on_click'):
        stop_button.on_click(
            lambda b: _handle_stop_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Stop button handler registered")

def _handle_stop_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Handle stop button click."""
    ui_components['stop_requested'] = True
    logger = ui_components.get('logger')
    if logger:
        logger.warning("🛑 Stop requested by user")