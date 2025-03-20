"""
File: smartcash/ui/dataset/split_config_handler.py
Deskripsi: Handler utama untuk konfigurasi split dataset dengan visualisasi yang disederhanakan
"""

from typing import Dict, Any, Optional
import logging
from IPython.display import display, HTML

def setup_split_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logging terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "split_config", log_level=logging.INFO)
        if logger:
            ui_components['logger'] = logger
            logger.info(f"🚀 Split config handler diinisialisasi")
    except ImportError:
        pass
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    try:
        # Register handlers untuk event UI
        from smartcash.ui.dataset.split_config_utils import register_event_handlers
        ui_components = register_event_handlers(ui_components, config, env, logger)
        
        # Initialize UI dengan data dari config
        from smartcash.ui.dataset.split_config_utils import initialize_from_config
        ui_components = initialize_from_config(ui_components, config, env, logger)
        
        # Mendapatkan dan menampilkan statistik dataset
        from smartcash.ui.dataset.split_config_visualization import load_and_display_dataset_stats
        load_and_display_dataset_stats(ui_components, config, env, logger)
        
    except Exception as e:
        # Tampilkan error sederhana
        if logger:
            logger.error(f"❌ Error saat setup handler: {str(e)}")
        
        if 'output_box' in ui_components:
            with ui_components['output_box']:
                from smartcash.ui.utils.constants import ICONS
                display(HTML(f"""
                <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px;">
                    <p>{ICONS.get('error', '❌')} Error saat setup handler: {str(e)}</p>
                </div>
                """))
    
    return ui_components