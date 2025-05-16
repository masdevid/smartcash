"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer_v2.py
Deskripsi: Initializer untuk modul augmentasi dataset dengan pendekatan DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
from smartcash.ui.dataset.augmentation.handlers.config_handler import update_config_from_ui, update_ui_from_config
from smartcash.ui.dataset.augmentation.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.augmentation.handlers.status_handler import update_augmentation_info
from smartcash.ui.dataset.augmentation.handlers.persistence_handler import ensure_ui_persistence

def setup_augmentation_handlers(ui_components: Dict[str, Any], env: Any, config: Any) -> Dict[str, Any]:
    """Setup handler spesifik untuk modul augmentasi"""
    # Setup button handlers
    ui_components = setup_button_handlers(ui_components, env, config)
    
    # Tambahkan referensi ke handler
    ui_components['update_config_from_ui'] = update_config_from_ui
    ui_components['update_ui_from_config'] = update_ui_from_config
    
    # Pastikan UI persisten
    ensure_ui_persistence(ui_components)
    
    # Update informasi augmentasi
    update_augmentation_info(ui_components)
    
    return ui_components

def setup_augmentation_config_handler(ui_components: Dict[str, Any], config: Any, env: Any) -> Dict[str, Any]:
    """Setup config handler untuk modul augmentasi"""
    # Update UI dari konfigurasi
    if 'update_ui_from_config' in ui_components and callable(ui_components['update_ui_from_config']):
        ui_components['update_ui_from_config'](ui_components, config)
    
    return ui_components

def detect_augmentation_state(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Deteksi state augmentasi"""
    # Implementasi deteksi state jika diperlukan
    return ui_components

def initialize_augmentation_ui() -> Dict[str, Any]:
    """Inisialisasi UI modul augmentasi dataset."""
    
    # Konfigurasi multi-progress tracking
    multi_progress_config = {
        "module_name": "augmentation",
        "step_key": "augmentation_step",
        "progress_bar_key": "progress_bar",
        "current_progress_key": "current_progress",
        "overall_label_key": "overall_label",
        "step_label_key": "step_label"
    }
    
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['augment_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button']
    
    # Gunakan base initializer
    ui_components = initialize_module_ui(
        module_name='augmentation',
        create_ui_func=create_augmentation_ui,
        setup_config_handler_func=setup_augmentation_config_handler,
        setup_specific_handlers_func=setup_augmentation_handlers,
        detect_state_func=detect_augmentation_state,
        button_keys=button_keys,
        multi_progress_config=multi_progress_config,
        observer_group="augmentation_observers"
    )
    
    # Alihkan output log console ke UI
    from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
    intercept_stdout_to_ui(ui_components)
    
    # Tambahkan direct UI logger untuk integrasi yang lebih baik
    from smartcash.ui.utils.ui_logger import create_direct_ui_logger
    ui_logger = create_direct_ui_logger(ui_components, name="augmentation")
    ui_components['ui_logger'] = ui_logger
    
    # Log pesan bahwa log telah dialihkan
    ui_logger.info("🔄 Output log console telah dialihkan ke UI")
    
    return ui_components
