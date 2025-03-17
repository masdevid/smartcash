"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Thread pool untuk operasi asinkron
_executor = ThreadPoolExecutor(max_workers=2)

def setup_environment_config():
    """Setup dan jalankan konfigurasi environment"""
    
    # Import komponen
    from smartcash.ui.setup.env_config_component import create_env_config_ui
    from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
    
    try:
        # Coba mendapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
    except ImportError:
        env = None
    
    try:
        # Coba membaca konfigurasi
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        config = config_manager.config
    except ImportError:
        config = {}
    
    # Buat komponen UI
    ui_components = create_env_config_ui(env, config)
    
    # Setup handlers
    ui_components = setup_env_config_handlers(ui_components, env, config, _executor)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components