"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Inisialisasi UI untuk dataset downloader
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.components import create_download_ui
from smartcash.ui.utils.ui_logger import log_to_ui

def initialize_dataset_download_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset downloader.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    try:
        # Get config manager dengan fallback otomatis
        config_manager = get_config_manager()
        
        # Get base config
        base_config = config_manager.config
        
        # Merge dengan config yang diberikan
        if config:
            base_config.update(config)
            
        # Create UI components
        ui_components = create_download_ui(base_config)
        
        # Setup handlers
        ui_components = setup_download_handlers(ui_components, env, base_config)
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        log_to_ui(None, f"❌ Error saat inisialisasi UI: {str(e)}", "error")
        raise