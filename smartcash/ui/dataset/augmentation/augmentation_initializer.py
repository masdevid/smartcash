"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Inisialisasi antarmuka augmentasi dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger

def initialize_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Inisialisasi antarmuka augmentasi dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI
    """
    logger = get_logger('augmentation')
    
    # Import komponen UI
    from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
    
    # Buat komponen UI
    ui_components = create_augmentation_ui(env, config)
    
    # Tambahkan logger
    ui_components['logger'] = logger
    
    # Setup handler
    setup_handlers(ui_components, env, config)
    
    # Inisialisasi UI dari konfigurasi
    if 'update_ui_from_config' in ui_components and callable(ui_components['update_ui_from_config']):
        ui_components['update_ui_from_config'](ui_components)
    
    # Pastikan UI persisten
    from smartcash.ui.dataset.augmentation.handlers.persistence_handler import ensure_ui_persistence
    ensure_ui_persistence(ui_components)
    
    # Update informasi augmentasi
    from smartcash.ui.dataset.augmentation.handlers.status_handler import update_augmentation_info
    update_augmentation_info(ui_components)
    
    return ui_components

def setup_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk antarmuka augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang diupdate
    """
    # Import handler
    from smartcash.ui.dataset.augmentation.handlers.button_handlers import setup_button_handlers
    from smartcash.ui.dataset.augmentation.handlers.config_handler import update_config_from_ui, update_ui_from_config
    from smartcash.ui.dataset.augmentation.handlers.initialization_handler import register_progress_callback, reset_progress_bar
    
    # Setup handler
    ui_components = setup_button_handlers(ui_components, env, config)
    
    # Tambahkan referensi ke handler
    ui_components['update_config_from_ui'] = update_config_from_ui
    ui_components['update_ui_from_config'] = update_ui_from_config
    ui_components['register_progress_callback'] = register_progress_callback
    ui_components['reset_progress_bar'] = reset_progress_bar
    
    return ui_components

def display_augmentation_ui(ui_components: Dict[str, Any]) -> None:
    """
    Tampilkan antarmuka augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Tampilkan UI
    display(ui_components['ui'])

def create_and_display_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat dan tampilkan antarmuka augmentasi dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI
    """
    # Inisialisasi UI
    ui_components = initialize_augmentation_ui(env, config)
    
    # Tampilkan UI
    display_augmentation_ui(ui_components)
    
    return ui_components
