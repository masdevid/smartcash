"""
File: smartcash/ui/dataset/augmentation/handlers/persistence_handler.py
Deskripsi: Handler untuk persistensi konfigurasi augmentation dataset
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.augmentation.handlers.config_handler import get_config_from_ui, update_config_from_ui

logger = get_logger(__name__)

def ensure_ui_persistence(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memastikan UI components memiliki fungsi persistensi yang diperlukan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah ditambahkan fungsi persistensi
    """
    # Tambahkan fungsi get_augmentation_config jika belum ada
    if 'get_augmentation_config' not in ui_components:
        ui_components['get_augmentation_config'] = get_config_from_ui
    
    # Tambahkan fungsi sync_config_with_drive jika belum ada
    if 'sync_config_with_drive' not in ui_components:
        ui_components['sync_config_with_drive'] = lambda: None
    
    # Tambahkan fungsi reset_config_to_default jika belum ada
    if 'reset_config_to_default' not in ui_components:
        ui_components['reset_config_to_default'] = lambda: None
    
    # Tambahkan pemanggilan register_ui_components
    config_manager = get_config_manager()
    config_manager.register_ui_components('augmentation', ui_components)
    
    return ui_components

def setup_persistence_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk persistensi konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah ditambahkan handler persistensi
    """
    # Pastikan UI components memiliki fungsi persistensi
    ui_components = ensure_ui_persistence(ui_components)
    
    # Tambahkan handler untuk save button jika ada
    if 'save_button' in ui_components:
        def on_save_click(b):
            config = get_config_from_ui(ui_components)
            update_config_from_ui(ui_components)
            logger.info("✅ Konfigurasi berhasil disimpan")
        
        ui_components['save_button'].on_click(on_save_click)
    
    return ui_components 