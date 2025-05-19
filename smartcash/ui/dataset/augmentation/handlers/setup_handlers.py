"""
File: smartcash/ui/dataset/augmentation/handlers/setup_handlers.py
Deskripsi: Setup handlers untuk modul augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger("augmentation_setup")

def setup_augmentation_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handlers untuk augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Tambahkan logger ke ui_components jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = logger
        
    # Import handlers
    try:
        from smartcash.ui.dataset.augmentation.handlers.status_handler import setup_status_handler
        from smartcash.ui.dataset.augmentation.handlers.observer_handler import setup_observer_handler
        from smartcash.ui.dataset.augmentation.handlers.state_handler import setup_state_handler
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import setup_persistence_handler
        from smartcash.ui.dataset.augmentation.handlers.button_handler import setup_augmentation_button_handlers
        
        # Setup status handler
        ui_components = setup_status_handler(ui_components)
        
        # Setup observer handler
        ui_components = setup_observer_handler(ui_components)
        
        # Setup state handler
        ui_components = setup_state_handler(ui_components, env, config)
        
        # Setup persistence handler
        ui_components = setup_persistence_handler(ui_components, env, config)
        
        # Setup button handlers
        ui_components = setup_augmentation_button_handlers(ui_components, config=config, env=env)
        
        # Setup augmentation service
        from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import initialize_augmentation
        ui_components = initialize_augmentation(ui_components)
        
        logger.info(f"{ICONS['success']} Handler augmentasi berhasil disetup")
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat setup handler augmentasi: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Log hasil setup
    logger.debug(f"{ICONS['info']} Semua handler augmentasi berhasil disetup")
    
    return ui_components

def setup_state_handler(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handler untuk state augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state
    
    # Deteksi state augmentasi
    ui_components = detect_augmentation_state(ui_components)
    
    return ui_components
