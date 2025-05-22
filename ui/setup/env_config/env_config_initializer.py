"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Inisialisasi UI untuk konfigurasi environment - disederhanakan sebagai entry point
"""

from typing import Dict, Any
import logging

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi konfigurasi environment dengan error handling yang proper
    
    Returns:
        Dictionary UI components
    """
    try:
        # Import component
        from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        
        # Buat dan tampilkan component
        env_config = EnvConfigComponent()
        env_config.display()
        
        # Return ui_components untuk kompatibilitas
        return env_config.ui_components
        
    except Exception as e:
        # Log error dengan proper logger
        logger = logging.getLogger('smartcash.ui.setup.env_config')
        logger.error(f"Error initializing environment config UI: {str(e)}")
        
        # Create minimal error UI sebagai fallback
        try:
            from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
            error_components = UIFactory.create_error_ui_components(
                f"Gagal menginisialisasi environment config: {str(e)}"
            )
            
            # Display error UI
            from IPython.display import display
            display(error_components['ui_layout'])
            
            return error_components
            
        except Exception as fallback_error:
            # Ultimate fallback jika semua gagal
            logger.error(f"Fallback error UI juga gagal: {str(fallback_error)}")
            return {
                'error': str(e),
                'fallback_error': str(fallback_error)
            }


# Alias untuk kompatibilitas mundur
def initialize_environment_config_ui() -> Dict[str, Any]:
    """Alias untuk kompatibilitas mundur"""
    return initialize_env_config_ui()