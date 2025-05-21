"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Inisialisasi UI untuk konfigurasi environment
"""

from typing import Dict, Any
import logging

from smartcash.ui.setup.env_config.components import EnvConfigComponent, UIFactory

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi konfigurasi environment
    
    Returns:
        Dictionary UI components
    """
    try:
        # Buat EnvConfigComponent
        env_config = EnvConfigComponent()
        
        # Tampilkan UI
        env_config.display()
        
        # Return ui_components
        return env_config.ui_components
        
    except Exception as e:
        # Log error
        logger = logging.getLogger('smartcash.ui.setup.env_config')
        logger.error(f"Error initializing environment config UI: {str(e)}")
        
        # Create error UI
        return UIFactory.create_error_ui_components(str(e))
