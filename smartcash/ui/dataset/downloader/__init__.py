"""
File: smartcash/ui/dataset/downloader/__init__.py
Deskripsi: Updated entry point dengan streamlined exports dan backward compatibility
"""

# Main initialization (streamlined)
from .downloader_init import (
    initialize_downloader_ui, 
    get_downloader_status,
    get_downloader_config,
    update_downloader_config,
    validate_downloader_setup
)

# UI Components (preserved interface)
from .components.ui_layout import create_downloader_ui
from .components.ui_form import create_form_fields

# New streamlined handlers
from .handlers.config_handler import DownloaderConfigHandler
from .handlers.streamlined_download_handler import setup_streamlined_download_handlers
from .handlers.progress_integration import create_progress_integrator

# Configuration (preserved)
from .handlers.defaults import DEFAULT_CONFIG, get_default_downloader_config

__all__ = [
    # Main API (streamlined)
    'initialize_downloader_ui', 'get_downloader_status', 'setup_downloader',
    
    # Configuration management
    'get_downloader_config', 'update_downloader_config', 'validate_downloader_setup',
    
    # UI Components (preserved interface)
    'create_downloader_ui', 'create_form_fields',
    
    # Handlers (streamlined)
    'DownloaderConfigHandler', 'setup_streamlined_download_handlers', 'create_progress_integrator',
    
    # Configuration
    'DEFAULT_CONFIG', 'get_default_downloader_config'
]

# Public API untuk easy integration - main entry point
def setup_downloader(env=None, config=None, **kwargs):
    """
    Setup downloader UI dengan streamlined configuration.
    
    Args:
        env: Environment manager instance
        config: Optional config override
        **kwargs: Additional parameters
        
    Returns:
        UI components dengan integrated handlers
    """
    return initialize_downloader_ui(env=env, config=config, **kwargs)

def get_status():
    """Get downloader status untuk debugging dan validation."""
    return get_downloader_status()

# Backward compatibility aliases (to maintain existing cell imports)
from .downloader_init import (
    extract_downloader_config,  # backward compatibility
    update_downloader_ui,       # backward compatibility  
    setup_download_handlers     # backward compatibility (redirects to streamlined)
)

# Export backward compatibility functions
__all__.extend([
    'extract_downloader_config',
    'update_downloader_ui', 
    'setup_download_handlers'
])