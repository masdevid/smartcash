"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer untuk modul preprocessing dataset
"""

from typing import Dict, Any
from IPython.display import display

def initialize_preprocessing_ui() -> Dict[str, Any]:
    """Inisialisasi UI modul preprocessing dataset."""
    ui_components = {'module_name': 'preprocessing'}
    
    try:
        # Setup environment dan config (gunakan singleton)
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        env, config = setup_notebook_environment('preprocessing')
        
        # Buat komponen UI
        from smartcash.ui.dataset.preprocessing.components.preprocessing_component import create_preprocessing_ui
        ui_components = create_preprocessing_ui(env, config)
        
        # Setup logging untuk UI
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, 'preprocessing')
        ui_components['logger'] = logger
        
        # Setup config handler dengan pendekatan granular
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_preprocessing_config_handler
        ui_components = setup_preprocessing_config_handler(ui_components, config, env)
        
        # Setup progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(
            ui_components, 
            "preprocessing", 
            "preprocessing_step"
        )
        
        # Setup button handlers dengan pendekatan granular
        from smartcash.ui.dataset.preprocessing.handlers.button_handlers import setup_button_handlers
        ui_components = setup_button_handlers(ui_components, env, config)
        
        # Setup visualization dan cleanup handlers
        from smartcash.ui.dataset.preprocessing.handlers.visualization_handler import setup_visualization_handlers
        ui_components = setup_visualization_handlers(ui_components, env, config)
        
        from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import setup_cleanup_handler
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Setup observer integration
        try:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components, "preprocessing_observers")
        except ImportError:
            pass
        
        # Deteksi status data preprocessing
        from smartcash.ui.dataset.preprocessing.handlers.state_handler import detect_preprocessing_state
        ui_components = detect_preprocessing_state(ui_components)
        
        # Register cleanup function untuk cell execution
        from smartcash.ui.utils.logging_utils import register_cleanup_on_cell_execution
        register_cleanup_on_cell_execution(ui_components)
        
        if logger: logger.info("🚀 Preprocessing UI berhasil diinisialisasi")
        
        # Tampilkan UI
        display(ui_components['ui'])
        
    except Exception as e:
        # Fallback jika ada error
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, f"⚠️ Error saat inisialisasi preprocessing: {str(e)}", "error")
        display(ui_components['ui'])
    
    return ui_components