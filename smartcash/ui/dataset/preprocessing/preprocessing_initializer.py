"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer untuk modul preprocessing dataset dengan pendekatan DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.preprocessing.components.preprocessing_component import create_preprocessing_ui
from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_preprocessing_config_handler
from smartcash.ui.dataset.preprocessing.handlers.state_handler import detect_preprocessing_state
# Import visualization handler dari modul preprocessing
from smartcash.ui.dataset.preprocessing.handlers.visualization_handler import setup_visualization_handlers

def initialize_preprocessing_ui() -> Dict[str, Any]:
    """Inisialisasi UI modul preprocessing dataset."""
    
    # Konfigurasi multi-progress tracking
    multi_progress_config = {
        "module_name": "preprocessing",
        "step_key": "preprocessing_step",
        "progress_bar_key": "progress_bar",
        "current_progress_key": "current_progress",
        "overall_label_key": "overall_label",
        "step_label_key": "step_label"
    }
    
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['preprocess_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button']
    
    # Gunakan base initializer
    return initialize_module_ui(
        module_name='preprocessing',
        create_ui_func=create_preprocessing_ui,
        setup_config_handler_func=setup_preprocessing_config_handler,
        setup_specific_handlers_func=lambda ui_components, env, config: setup_visualization_handlers(ui_components, env=env, config=config),
        detect_state_func=detect_preprocessing_state,
        button_keys=button_keys,
        multi_progress_config=multi_progress_config,
        observer_group="preprocessing_observers"
    )