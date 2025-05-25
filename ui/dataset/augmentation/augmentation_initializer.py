"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk UI augmentasi dataset dengan integrasi shared components terbaru
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager

# Konstanta untuk namespace logger
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE

# Import handlers
from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_augmentation_button_click
from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import handle_cleanup_button_click
from smartcash.ui.dataset.augmentation.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.augmentation.handlers.save_handler import handle_save_button_click

# Import utils
from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers

# Import komponen UI
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui

# Flag global untuk mencegah inisialisasi ulang
_AUGMENTATION_MODULE_INITIALIZED = False

def initialize_dataset_augmentation_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset augmentation dengan shared components.
    
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
        logger.info("🚀 Memulai inisialisasi UI augmentasi dataset dengan shared components")
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
            
        # Create UI components dengan shared components
        ui_components = create_augmentation_ui(env, augmentation_config)
        
        # Setup logger dengan bridge pattern untuk mencegah circular dependency
        ui_logger = create_ui_logger_bridge(ui_components, AUGMENTATION_LOGGER_NAMESPACE)
        ui_components['logger'] = ui_logger
        ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
        ui_components['augmentation_initialized'] = True
        
        # Setup shared button state manager
        button_state_manager = get_button_state_manager(ui_components)
        ui_components['button_state_manager'] = button_state_manager
        
        # Setup handlers dengan shared components
        ui_components = setup_augmentation_handlers(ui_components, env, augmentation_config)
        
        logger.success("✅ UI augmentasi dataset berhasil diinisialisasi dengan shared components")
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        raise

def setup_augmentation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI augmentasi dataset dengan shared components.
    
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
    
    ui_logger = ui_components['logger']
    
    # Setup shared button state manager
    if 'button_state_manager' not in ui_components:
        button_state_manager = get_button_state_manager(ui_components)
        ui_components['button_state_manager'] = button_state_manager
    
    # Setup observer untuk notifikasi
    try:
        observer_manager = register_ui_observers(ui_components)
        ui_components['observer_manager'] = observer_manager
    except Exception as e:
        ui_logger.warning(f"Gagal setup observer: {str(e)}")
    
    # Setup handlers untuk UI events dengan shared components
    _setup_button_handlers(ui_components)
    
    # Setup shared progress tracking
    try:
        _setup_shared_progress_tracking(ui_components)
    except Exception as e:
        ui_logger.warning(f"Error setup progress: {str(e)}")
    
    # Save config ke UI components
    ui_components['config'] = config or {}
    
    # Load konfigurasi dan update UI jika ada
    try:
        if config and isinstance(config, dict):
            from smartcash.ui.dataset.augmentation.handlers.config_handler import update_ui_from_config
            update_ui_from_config(ui_components, config)
    except Exception as e:
        ui_logger.warning(f"Gagal update UI dari config: {str(e)}")
    
    ui_logger.success("✅ Augmentasi dataset handlers berhasil diinisialisasi dengan shared components")
    
    return ui_components

def _setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol UI dengan safe checking dan shared components."""
    
    logger = ui_components.get('logger')
    
    # Debug: Log available button keys
    if logger:
        available_buttons = [k for k in ui_components.keys() if 'button' in k and ui_components[k] is not None]
        logger.debug(f"🔍 Available buttons: {available_buttons}")
    
    # Setup augmentation button - Safe checking
    augment_button = ui_components.get('augment_button')
    if augment_button is not None and hasattr(augment_button, 'on_click'):
        augment_button.on_click(
            lambda b: handle_augmentation_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Augment button handler registered dengan shared components")
    else:
        if logger:
            logger.warning("⚠️ Augment button tidak ditemukan")
    
    # Setup cleanup button - Safe checking
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button is not None and hasattr(cleanup_button, 'on_click'):
        cleanup_button.on_click(
            lambda b: handle_cleanup_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Cleanup button handler registered dengan shared components")
    else:
        if logger:
            logger.warning("⚠️ Cleanup button tidak ditemukan")
    
    # Setup reset button - Safe checking
    reset_button = ui_components.get('reset_button')
    if reset_button is not None and hasattr(reset_button, 'on_click'):
        reset_button.on_click(
            lambda b: handle_reset_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Reset button handler registered dengan shared components")
    else:
        if logger:
            logger.warning("⚠️ Reset button tidak ditemukan")
    
    # Setup save button - Safe checking
    save_button = ui_components.get('save_button')
    if save_button is not None and hasattr(save_button, 'on_click'):
        save_button.on_click(
            lambda b: handle_save_button_click(ui_components, b)
        )
        if logger:
            logger.debug("✅ Save button handler registered dengan shared components")
    else:
        if logger:
            logger.warning("⚠️ Save button tidak ditemukan")

def _setup_shared_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup shared progress tracking untuk augmentasi."""
    
    # Cek apakah shared progress tracking sudah tersedia
    if 'tracker' in ui_components:
        ui_components['logger'].debug("✅ Shared progress tracking sudah tersedia")
        return
    
    # Setup fallback progress functions jika shared component belum tersedia
    if not callable(ui_components.get('update_progress')):
        def fallback_update_progress(progress_type: str, value: int, message: str = ""):
            # Fallback untuk backward compatibility
            if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
                ui_components['progress_bar'].value = value
                ui_components['progress_bar'].description = f"Progress: {value}%"
            
            # Update message labels
            if message:
                for label_key in ['progress_message', 'step_label', 'overall_label']:
                    if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
                        ui_components[label_key].value = message
        
        ui_components['update_progress'] = fallback_update_progress
    
    if not callable(ui_components.get('reset_all')):
        def fallback_reset_progress():
            if 'progress_bar' in ui_components:
                if hasattr(ui_components['progress_bar'], 'value'):
                    ui_components['progress_bar'].value = 0
                if hasattr(ui_components['progress_bar'], 'layout'):
                    ui_components['progress_bar'].layout.visibility = 'hidden'
        
        ui_components['reset_all'] = fallback_reset_progress
    
    ui_components['logger'].debug("✅ Fallback progress tracking berhasil disetup")