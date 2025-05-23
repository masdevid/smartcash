"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer ultra sederhana untuk preprocessing
"""

from typing import Any
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager

def initialize_dataset_preprocessing_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI preprocessing sederhana.
    
    Returns:
        Widget UI utama
    """
    logger = get_logger("smartcash.ui.dataset.preprocessing")
    
    try:
        # 1. Setup basic components
        ui_components = {
            'config_manager': get_config_manager(),
            'data_dir': 'data',
            'preprocessed_dir': 'data/preprocessed',
            'logger': logger,
            'preprocessing_running': False,
            'stop_requested': False
        }
        
        # 2. Create UI
        from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
        ui_components.update(create_preprocessing_main_ui(config))
        
        # 3. Setup handlers
        from smartcash.ui.dataset.preprocessing.handlers.main_handler import setup_main_handler
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_handlers
        from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import setup_cleanup_handler
        
        setup_main_handler(ui_components)
        setup_config_handlers(ui_components)
        setup_cleanup_handler(ui_components)
        
        logger.success("✅ UI preprocessing berhasil diinisialisasi")
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"❌ Error inisialisasi: {str(e)}")
        return _create_error_ui(str(e))

def _create_error_ui(error_message: str) -> Any:
    """Buat UI error minimal."""
    import ipywidgets as widgets
    
    return widgets.HTML(f"""
    <div style="padding: 20px; background-color: #f8d7da; border-radius: 5px; color: #721c24;">
        <h3>❌ Error Preprocessing</h3>
        <p>{error_message}</p>
        <p>Restart kernel dan coba lagi.</p>
    </div>
    """)

# Alias
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui