"""
File: smartcash/ui/dataset/downloader/handlers/__init__.py
Deskripsi: Updated handlers export dengan unified handler approach
"""

# Unified handler exports
from .download_handler import setup_download_handlers, UnifiedDownloadHandler
from .config_handler import DownloaderConfigHandler, create_downloader_config_handler
from .defaults import (
    get_default_downloader_config, get_roboflow_defaults, 
    get_download_defaults, get_uuid_defaults
)
from .validation_handler import setup_validation_handler, validate_complete_form

# Backward compatibility - unified handler handles all operations
setup_check_handler = setup_download_handlers  # Same unified handler
setup_cleanup_handler = setup_download_handlers  # Same unified handler

# Export all
__all__ = [
    # Main handlers
    'setup_download_handlers',
    'UnifiedDownloadHandler',
    
    # Config
    'DownloaderConfigHandler', 
    'create_downloader_config_handler',
    
    # Defaults
    'get_default_downloader_config',
    'get_roboflow_defaults',
    'get_download_defaults', 
    'get_uuid_defaults',
    
    # Validation
    'setup_validation_handler',
    'validate_complete_form',
    
    # Backward compatibility
    'setup_check_handler',
    'setup_cleanup_handler'
]