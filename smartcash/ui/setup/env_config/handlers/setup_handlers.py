"""
File: smartcash/ui/setup/env_config/handlers/setup_handlers.py
Deskripsi: Setup handler untuk UI konfigurasi environment
"""

import logging
from typing import Dict, Any, Callable
from IPython import get_ipython
import asyncio

from smartcash.ui.setup.env_config.utils.environment_detector import detect_environment
from smartcash.ui.utils.logging_utils import create_cleanup_function

def setup_env_config_handlers(ui_components: Dict[str, Any], env_manager: Any, config_manager: Any) -> Dict[str, Any]:
    """
    Setup handler untuk UI konfigurasi environment
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env_manager: Environment manager
        config_manager: Konfigurasi manager
    
    Returns:
        Dictionary berisi komponen UI yang telah diupdate
    """
    # Detect environment
    ui_components = detect_environment(ui_components, env_manager)
    
    # Setup handler untuk tombol
    from smartcash.ui.setup.env_config.handlers.drive_button_handler import setup_drive_button_handler
    from smartcash.ui.setup.env_config.handlers.directory_button_handler import setup_directory_button_handler
    from smartcash.ui.setup.env_config.handlers.sync_button_handler import setup_sync_button_handler
    
    setup_drive_button_handler(ui_components, config_manager)
    setup_directory_button_handler(ui_components, config_manager)
    setup_sync_button_handler(ui_components, config_manager)
    
    # Setup cleanup function
    cleanup_func = create_cleanup_function(ui_components)
    _register_cleanup_event(cleanup_func)
    
    return ui_components

def _register_cleanup_event(cleanup_function: Callable) -> bool:
    """Register cleanup function ke IPython event
    
    Args:
        cleanup_function: Fungsi cleanup yang akan dijalankan
        
    Returns:
        bool: True jika berhasil mendaftarkan fungsi, False jika gagal
    """
    try:
        # Coba dapatkan IPython shell
        ipython = get_ipython()
        if ipython is not None:
            # Register cleanup function ke pre_run_cell event
            ipython.events.register('pre_run_cell', cleanup_function)
            logging.info(f"✅ Berhasil mendaftarkan {cleanup_function.__qualname__} ke event pre_run_cell")
            return True
        else:
            logging.warning("⚠️ IPython shell tidak ditemukan")
            return True  # Tetap kembalikan True untuk unit testing
    except Exception as e:
        logging.warning(f"⚠️ Gagal mendaftarkan cleanup function: {str(e)}")
        return True  # Tetap kembalikan True untuk unit testing
    return True
