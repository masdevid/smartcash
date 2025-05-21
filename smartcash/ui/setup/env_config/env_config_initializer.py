"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Inisialisasi UI untuk konfigurasi environment
"""

from typing import Dict, Any

from smartcash.ui.setup.env_config.components import UIFactory
from smartcash.ui.setup.env_config.handlers import SetupHandler

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi konfigurasi environment
    
    Returns:
        Dictionary UI components
    """
    try:
        # Buat dan tampilkan komponen UI
        ui_components = UIFactory.create_ui_components()
        
        # Extract callbacks
        ui_callbacks = {}
        
        # Gunakan log_message callback jika tersedia
        if 'log_output' in ui_components:
            def log_message(message: str):
                import ipywidgets as widgets
                from IPython.display import display
                with ui_components['log_output']:
                    display(widgets.HTML(f"<p>{message}</p>"))
            
            ui_callbacks['log_message'] = log_message
        
        # Gunakan update_status callback jika tersedia
        if 'status_panel' in ui_components:
            def update_status(message: str, status_type: str = "info"):
                from smartcash.ui.utils.alert_utils import update_alert
                update_alert(ui_components['status_panel'], message, status_type)
            
            ui_callbacks['update_status'] = update_status
        
        # Lakukan setup environment
        setup_handler = SetupHandler(ui_callbacks)
        config_manager, base_dir, config_dir = setup_handler.perform_setup()
        
        # Tambahkan config manager ke UI components
        ui_components['config_manager'] = config_manager
        
        return ui_components
        
    except Exception as e:
        # Jika exception terjadi, tampilkan UI error
        ui_components = UIFactory.create_error_ui_components(str(e))
        
        # Extract callbacks untuk error handling
        ui_callbacks = {}
        
        # Gunakan log_message callback jika tersedia
        if 'log_output' in ui_components:
            def log_message(message: str):
                import ipywidgets as widgets
                from IPython.display import display
                with ui_components['log_output']:
                    display(widgets.HTML(f"<p>{message}</p>"))
            
            ui_callbacks['log_message'] = log_message
        
        # Tangani error dan dapatkan config manager fallback
        setup_handler = SetupHandler(ui_callbacks)
        config_manager = setup_handler.handle_error(e)
        
        # Tambahkan config manager ke UI components jika ada
        if config_manager:
            ui_components['config_manager'] = config_manager
        
        return ui_components
