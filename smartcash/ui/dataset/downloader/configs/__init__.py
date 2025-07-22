"""
Downloader configuration management.

This module provides configuration management for the dataset downloader module,
including default configurations and config handler classes.
"""

from typing import Dict, Any, Optional, List

from .downloader_config_handler import (
    DownloaderConfigHandler,
    get_downloader_config_handler
)
from .downloader_defaults import (
    get_default_downloader_config,
    get_default_config,
    get_roboflow_defaults,
    get_download_defaults,
    get_uuid_defaults,
    get_preset_workspaces,
    get_supported_formats,
    get_naming_strategies,
    get_default_workspace,
    get_default_project,
    get_default_version,
    is_uuid_enabled_by_default,
    is_validation_enabled_by_default
)

__all__ = [
    # Main config handler
    'DownloaderConfigHandler',
    'get_downloader_config_handler',
    
    # Default configs
    'get_default_downloader_config',
    'get_default_config',  # Alias for backward compatibility
    
    # Partial configs
    'get_roboflow_defaults',
    'get_download_defaults',
    'get_uuid_defaults',
    
    # UI helpers
    'get_preset_workspaces',
    'get_supported_formats',
    'get_naming_strategies',
    
    # Quick access helpers
    'get_default_workspace',
    'get_default_project',
    'get_default_version',
    'is_uuid_enabled_by_default',
    'is_validation_enabled_by_default'
]