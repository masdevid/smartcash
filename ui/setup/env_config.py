"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan integrasi utils, helpers dan handlers
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi utilities"""
    
    # Import komponen dengan pendekatan konsolidasi
    from smartcash.ui.setup.env_config_component import create_env_config_ui
    from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component
    from smartcash.ui.utils.logging_utils import setup_ipython_logging
    
    try:
        # Setup notebook environment
        env, config = setup_notebook_environment("env_config")
        
        # Buat komponen UI dengan helpers
        ui_components = create_env_config_ui(env, config)
        
        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, "env_config")
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Modul environment config berhasil dimuat")
        
        # Setup handlers untuk UI
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import import_with_fallback, show_status
        
        # Fallback environment setup
        env = type('DummyEnv', (), {
            'is_colab': 'google.colab' in __import__('sys').modules,
            'base_dir': __import__('os').getcwd(),
            'is_drive_mounted': False,
        })
        config = {}
        
        # Buat UI components
        ui_components = create_env_config_ui(env, config)
        
        # Tampilkan pesan error
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components