"""
File: smartcash/ui/dataset/preprocessing/handlers/setup_handlers.py
Deskripsi: Setup handler untuk preprocessing dataset
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup semua handler untuk preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger())
    
    # Tambahkan logger ke ui_components jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = logger
    
    # Setup config handler
    from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_preprocessing_config_handler
    ui_components = setup_preprocessing_config_handler(ui_components, config, env)
    
    # Setup button handlers
    from smartcash.ui.dataset.preprocessing.handlers.button_handler import setup_preprocessing_button_handlers
    ui_components = setup_preprocessing_button_handlers(ui_components, 'preprocessing', config, env)
    
    # Setup state handler
    from smartcash.ui.dataset.preprocessing.handlers.state_handler import setup_state_handler
    ui_components = setup_state_handler(ui_components)
    
    # Setup service handler
    from smartcash.ui.dataset.preprocessing.handlers.preprocessing_service_handler import initialize_preprocessing
    ui_components = initialize_preprocessing(ui_components)
    
    # Setup observer handler
    from smartcash.ui.dataset.preprocessing.handlers.observer_handler import setup_observer_handler
    ui_components = setup_observer_handler(ui_components, "preprocessing_observers")
    
    # Setup status handler
    from smartcash.ui.dataset.preprocessing.handlers.status_handler import setup_status_handler
    ui_components = setup_status_handler(ui_components)
    
    # Setup cleanup handler
    from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import setup_cleanup_handler
    ui_components = setup_cleanup_handler(ui_components)
    
    # Setup persistence handler
    from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import setup_persistence_handler
    ui_components = setup_persistence_handler(ui_components)
    
    # Log
    logger.debug(f"{ICONS['info']} Semua handler preprocessing berhasil disetup")
    
    return ui_components

def setup_state_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk state preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.dataset.preprocessing.handlers.state_handler import detect_preprocessing_state
    
    # Deteksi state preprocessing
    ui_components = detect_preprocessing_state(ui_components)
    
    return ui_components
