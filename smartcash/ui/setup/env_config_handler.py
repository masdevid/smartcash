"""
File: smartcash/ui/setup/env_config_handler.py
Deskripsi: Handler untuk komponen UI konfigurasi environment dengan integrasi ui_handlers
"""

import os
import sys
from typing import Dict, Any, Optional

def setup_env_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI environment config dengan integrasi ui_handlers.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import komponen dari handlers untuk mengurangi duplikasi
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    from smartcash.ui.setup.env_detection import detect_environment
    from smartcash.ui.setup.drive_handler import handle_drive_connection
    from smartcash.ui.setup.directory_handler import handle_directory_setup
    
    # Setup observer handlers untuk menerima event notifikasi
    ui_components = setup_observer_handlers(ui_components, "env_config_observers")
    
    # Deteksi environment jika belum ada
    ui_components = detect_environment(ui_components, env)
    
    # Decorator untuk try-except pada handler
    def try_except_decorator(handler_func):
        def wrapper(b):
            try:
                return handler_func(b)
            except Exception as e:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                with ui_components['status']:
                    display(create_status_indicator('error', f"❌ Error: {str(e)}"))
                if 'logger' in ui_components and ui_components['logger']:
                    ui_components['logger'].error(f"❌ Error dalam handler: {str(e)}")
        return wrapper
    
    # Handler untuk tombol Drive dengan decorator error handling
    @try_except_decorator
    def on_drive_button_clicked(b):
        handle_drive_connection(ui_components)
    
    # Handler untuk tombol Directory dengan decorator error handling
    @try_except_decorator
    def on_directory_button_clicked(b):
        handle_directory_setup(ui_components)
    
    # Daftarkan handler ke tombol
    ui_components['drive_button'].on_click(on_drive_button_clicked)
    ui_components['directory_button'].on_click(on_directory_button_clicked)
    
    # Register cleanup function jika ada resources yang perlu dibersihkan
    def cleanup_resources():
        """Bersihkan resources saat cell di-reset atau dieksekusi ulang"""
        if 'observer_group' in ui_components:
            try:
                from smartcash.components.observer.manager_observer import ObserverManager
                observer_manager = ObserverManager()
                observer_manager.unregister_group(ui_components['observer_group'])
            except ImportError:
                pass
    
    ui_components['cleanup'] = cleanup_resources
    
    return ui_components