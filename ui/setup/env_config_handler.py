"""
File: smartcash/ui/setup/env_config_handler.py
Deskripsi: Handler untuk komponen UI konfigurasi environment
"""

import os
import sys
from typing import Dict, Any, Optional

def setup_env_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI environment config.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.setup.env_detection import detect_environment
    from smartcash.ui.setup.drive_handler import handle_drive_connection
    from smartcash.ui.setup.directory_handler import handle_directory_setup
    
    # Deteksi environment jika belum ada
    ui_components = detect_environment(ui_components, env)
    
    # Handler untuk tombol Drive
    def on_drive_button_clicked(b):
        handle_drive_connection(ui_components)
    
    # Handler untuk tombol Directory
    def on_directory_button_clicked(b):
        handle_directory_setup(ui_components)
    
    # Daftarkan handler
    ui_components['drive_button'].on_click(on_drive_button_clicked)
    ui_components['directory_button'].on_click(on_directory_button_clicked)
    
    return ui_components