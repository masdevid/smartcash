"""
File: smartcash/ui/dataset/downloader/__init__.py  
Deskripsi: UI entry point untuk downloader module dengan clean public API
"""

from .downloader_init import initialize_downloader_ui, get_downloader_status
from .components.main_ui import create_downloader_ui
from .handlers.config_extractor import DownloaderConfigExtractor
from .handlers.config_updater import DownloaderConfigUpdater
from .handlers.defaults import DEFAULT_CONFIG
from .components.action_buttons import create_action_buttons, update_button_states
from .components.form_fields import create_form_fields

__all__ = [
    # Main initialization
    'initialize_downloader_ui', 'get_downloader_status',
    
    # UI Components  
    'create_downloader_ui', 'create_action_buttons', 'create_form_fields',
    
    # Config management
    'DownloaderConfigExtractor', 'DownloaderConfigUpdater', 'DEFAULT_CONFIG',
    
    # Utilities
    'update_button_states'
]

# Public API untuk easy integration dengan cells
def setup_downloader(env=None, config=None, **kwargs):
    """Setup downloader UI dengan minimal configuration - main entry point."""
    return initialize_downloader_ui(env=env, config=config, **kwargs)

def get_status():
    """Get downloader status untuk debugging."""
    return get_downloader_status()