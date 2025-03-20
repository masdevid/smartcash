"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash yang terintegrasi dengan tema dan komponen UI
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional

def setup_environment_config():
    """Setup dan jalankan konfigurasi environment dengan integrasi tema"""
    
    # Import komponen
    from smartcash.ui.setup.env_config_component import create_env_config_ui
    from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
    from smartcash.ui.helpers.ui_helpers import inject_css_styles
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    
    try:
        # Coba mendapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.logger import get_logger
        env = get_environment_manager()
        logger = get_logger("env_config")
    except ImportError:
        env = None
        logger = None
    
    try:
        # Coba membaca konfigurasi
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        config = config_manager.config
    except ImportError:
        config = {}
    
    # Inject CSS styles untuk konsistensi
    inject_css_styles()
    
    # Buat komponen UI
    ui_components = create_env_config_ui(env, config)
    
    # Setup handlers untuk error dan observer
    ui_components = setup_observer_handlers(ui_components, "env_config_observers")
    
    # Setup handlers spesifik untuk env config
    ui_components = setup_env_config_handlers(ui_components, env, config)
    
    # Tambahkan logger ke ui_components jika tersedia
    if logger:
        ui_components['logger'] = logger
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components