"""
File: smartcash/ui/dataset/downloader/handlers/__init__.py
Deskripsi: Handlers entry point dengan clean imports dan factories
"""

from .config_extractor import DownloaderConfigExtractor
from .config_updater import DownloaderConfigUpdater
from .download_handler import setup_download_handlers, setup_quick_action_handlers
from .progress_handler import ProgressCallbackManager, setup_progress_handlers, create_simple_progress_callback
from .validation_handler import validate_download_parameters, validate_workspace_project, get_validation_summary
from .defaults import DEFAULT_CONFIG, VALIDATION_RULES, get_default_api_key

__all__ = [
    # Config management
    'DownloaderConfigExtractor', 'DownloaderConfigUpdater', 'DEFAULT_CONFIG',
    
    # Handler setup
    'setup_download_handlers', 'setup_progress_handlers', 'setup_quick_action_handlers',
    
    # Progress management
    'ProgressCallbackManager', 'create_simple_progress_callback',
    
    # Validation
    'validate_download_parameters', 'validate_workspace_project', 'get_validation_summary',
    
    # Defaults
    'VALIDATION_RULES', 'get_default_api_key'
]