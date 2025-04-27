"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk modul augmentasi dataset dengan pendekatan DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
from smartcash.ui.dataset.augmentation.handlers.config_handler import setup_augmentation_config_handler
from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state
from smartcash.ui.handlers.visualization_handler import setup_visualization_handlers

def setup_augmentation_handlers(ui_components: Dict[str, Any], env: Any, config: Any) -> Dict[str, Any]:
    """Setup handler spesifik untuk modul augmentasi"""
    # Setup visualization handlers dengan shared handler
    return setup_visualization_handlers(ui_components, module_name='augmentation', env=env, config=config)

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
    return initialize_module_ui(
        module_name='augmentation',
        create_ui_func=create_augmentation_ui,
        setup_config_handler_func=setup_augmentation_config_handler,
        setup_specific_handlers_func=setup_augmentation_handlers,
        detect_state_func=detect_augmentation_state,
        button_keys=button_keys,
        multi_progress_config=multi_progress_config,
        observer_group="augmentation_observers"
    )