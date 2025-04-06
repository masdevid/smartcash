"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer untuk modul download dataset
"""

from typing import Dict, Any
from IPython.display import display

def initialize_dataset_download_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk download dataset.
    
    Returns:
        Dictionary UI components yang terinisialisasi
    """
    try:
        # Setup environment dan config
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        env = get_environment_manager()
        config = get_config_manager().config
        
        # Buat komponen UI
        from smartcash.ui.dataset.download.download_component import create_dataset_download_ui
        ui_components = create_dataset_download_ui(env, config)
        
        # Setup logging
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components)
        ui_components['logger'] = logger
        
        # Setup progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(
            ui_components, 
            overall_tracker_name="download", 
            step_tracker_name="download_step"
        )
        
        # Setup handlers
        from smartcash.ui.dataset.download.download_handlers import setup_download_handlers
        setup_download_handlers(ui_components, env, config)
        
        # Tampilkan UI
        display(ui_components['ui'])
        logger.info(f"✅ UI download dataset berhasil diinisialisasi")
        
        return ui_components
        
    except Exception as e:
        # Fallback minimal jika terjadi error
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        return create_fallback_ui({}, str(e), "error")