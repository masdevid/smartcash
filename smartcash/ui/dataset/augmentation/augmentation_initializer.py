"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk modul augmentasi dataset
"""

from typing import Dict, Any
from IPython.display import display, clear_output

def initialize_augmentation_ui() -> Dict[str, Any]:
    """Inisialisasi UI modul augmentasi dataset."""
    ui_components = {'module_name': 'augmentation'}
    
    try:
        # Setup environment dan config (gunakan singleton)
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        env, config = setup_notebook_environment('augmentation')
        
        # Setup logging untuk UI
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging({'module_name': 'augmentation'}, 'augmentation')
        ui_components['logger'] = logger
        
        # Buat komponen UI
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        ui_components = create_augmentation_ui(env, config)
        
        # Tambahkan logger ke ui_components
        ui_components['logger'] = logger
        
        # Setup config handler dengan pendekatan granular
        from smartcash.ui.dataset.augmentation.handlers.config_handler import setup_augmentation_config_handler
        ui_components = setup_augmentation_config_handler(ui_components, config, env)
        
        # Setup progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(
            ui_components, 
            "augmentation", 
            "augmentation_step", 
            "progress_bar", 
            "current_progress", 
            "overall_label", 
            "step_label"
        )
        
        # Setup button handlers dengan pendekatan granular
        from smartcash.ui.dataset.augmentation.handlers.button_handlers import setup_button_handlers
        ui_components = setup_button_handlers(ui_components, env, config)
        
        # Setup visualization dan cleanup handlers
        from smartcash.ui.dataset.augmentation.handlers.visualization_handler import setup_visualization_handlers
        ui_components = setup_visualization_handlers(ui_components, env, config)
        
        from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import setup_cleanup_handler
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Setup observer integration
        try:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components, "augmentation_observers")
            if logger: logger.debug("🔄 Observer berhasil diinisialisasi")
        except ImportError:
            pass
        
        # Deteksi status data augmentasi
        from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state
        ui_components = detect_augmentation_state(ui_components)
        
        # Register cleanup function untuk cell execution
        from smartcash.ui.utils.logging_utils import register_cleanup_on_cell_execution
        register_cleanup_on_cell_execution(ui_components)
        
        if logger: logger.info("🚀 Augmentation UI berhasil diinisialisasi")
        
    except Exception as e:
        # Fallback jika ada error
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, f"⚠️ Error saat inisialisasi augmentasi: {str(e)}", "error")
    
    return ui_components