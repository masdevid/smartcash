"""
File: smartcash/ui/training_config/hyperparameters/hyperparameters_initializer.py
Deskripsi: Initializer untuk UI konfigurasi hyperparameter model
"""

from typing import Dict, Any, Optional
from IPython.display import display

def initialize_hyperparameters_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi hyperparameter model.
    
    Args:
        env: Environment manager
        config: Konfigurasi untuk model
        
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {'module_name': 'hyperparameters'}
    
    try:
        # Import dependency
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        # Dapatkan environment dan config jika belum tersedia
        env = env or get_environment_manager()
        config = config or get_config_manager().config
        
        # Buat komponen UI
        from smartcash.ui.training_config.hyperparameters.components.hyperparameters_components import create_hyperparameters_ui
        ui_components.update(create_hyperparameters_ui(config))
        
        # Setup multi-progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(ui_components, "hyperparameters", "hyperparameters_step")
        
        # Setup handlers
        from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import setup_hyperparameters_button_handlers
        from smartcash.ui.training_config.hyperparameters.handlers.form_handlers import setup_hyperparameters_form_handlers
        
        # Setup handlers
        ui_components = setup_hyperparameters_button_handlers(ui_components, env, config)
        ui_components = setup_hyperparameters_form_handlers(ui_components, env, config)
        
        # Tampilkan container utama
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
        
    except Exception as e:
        # Gunakan utilitas fallback yang ada
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, str(e), "error")
    
    return ui_components
