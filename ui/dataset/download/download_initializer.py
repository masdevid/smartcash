"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Inisialisasi UI untuk download dataset
"""

import os
from pathlib import Path
from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager

from smartcash.ui.dataset.download.components.ui_creator import create_download_ui
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.handlers.config_handler import (
    load_config,
    save_config,
    update_config_from_ui,
    update_ui_from_config
)

logger = get_logger(__name__)

def initialize_dataset_download_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk download dataset.
    
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        # Setup environment dan config
        env_manager = get_environment_manager()
        base_dir = env_manager.base_dir
        
        # Initialize config manager with proper base directory
        config_manager = get_config_manager(
            base_dir=str(base_dir),
            config_file='dataset_config.yaml'
        )
        
        # Load konfigurasi
        config = load_config()
        
        # Buat komponen UI
        ui_components = create_download_ui()
        
        # Setup handlers
        ui_components = setup_download_handlers(ui_components)
        
        # Update UI dari konfigurasi
        update_ui_from_config(config, ui_components)
        
        # Register UI components dengan config manager
        config_manager.register_ui_components('dataset_download', ui_components)
        
        return ui_components
        
    except Exception as e:
        logger.error(f"⚠️ Error saat inisialisasi download: {str(e)}")
        raise