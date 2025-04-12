"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk modul augmentasi dataset
"""
from typing import Dict, Any

def initialize_augmentation_ui() -> Dict[str, Any]:
    """Inisialisasi UI modul augmentasi dataset."""
    ui_components = {'module_name': 'augmentation'}
    
    try:
        # Setup environment dan config
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        env, config = setup_notebook_environment('augmentation')
        
        # Buat komponen UI
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        ui_components = create_augmentation_ui(env, config)
        
        # Setup logging
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "augmentation")
        ui_components['logger'] = logger
        
        # Setup multi-progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(
            ui_components, 
            overall_tracker_name="augmentation", 
            step_tracker_name="augmentation_step"
        )
        
        # Setup handlers
        from smartcash.ui.dataset.augmentation.handlers.button_handlers import setup_button_handlers
        from smartcash.ui.dataset.augmentation.handlers.config_handlers import setup_config_handlers
        from smartcash.ui.dataset.augmentation.handlers.visualization_handlers import setup_visualization_handlers
        
        ui_components = setup_button_handlers(ui_components, env, config)
        ui_components = setup_config_handlers(ui_components, env, config)
        ui_components = setup_visualization_handlers(ui_components, env, config)
        
        # Setup status dan deteksi state
        from smartcash.ui.dataset.shared.setup_utils import detect_module_state
        ui_components = detect_module_state(ui_components, 'augmentation')
        
        # Setup AugmentationManager
        from smartcash.ui.dataset.shared.setup_utils import setup_manager
        augmentation_manager = setup_manager(ui_components, config, 'augmentation')
        ui_components['augmentation_manager'] = augmentation_manager
        
        # Setup observer dan event handling
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        ui_components = setup_observer_handlers(ui_components, "augmentation_observers")
        
        # Setup cleanup function
        from smartcash.ui.utils.logging_utils import create_cleanup_function
        cleanup_func = create_cleanup_function(ui_components)
        ui_components['cleanup'] = cleanup_func
        
        # Register cleanup to IPython events
        from IPython import get_ipython
        if get_ipython() and cleanup_func:
            get_ipython().events.register('pre_run_cell', cleanup_func)
        
        # Log success
        logger.info("🚀 Augmentation UI berhasil diinisialisasi")
        
    except Exception as e:
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, str(e), "error")
    
    return ui_components