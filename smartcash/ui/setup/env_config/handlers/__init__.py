"""
Package containing handlers for the env_config module.

This package provides various handlers that follow the CommonInitializer pattern
for managing different aspects of the environment configuration setup.
"""

# Import from main handlers directory
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler as BaseConfigHandler

# Import local handlers
from .config_handler import ConfigHandler
from .drive_handler import DriveHandler
from .error_handler import EnvConfigErrorHandler, with_error_handling, create_error_handler
from .folder_handler import FolderHandler
from .setup_handler import SetupHandler
from .status_checker import StatusChecker

__all__ = [
    'BaseHandler',  # Use BaseHandler instead of BaseEnvHandler
    'BaseConfigHandler',  # Add BaseConfigHandler for reference
    'ConfigHandler',
    'create_error_handler',
    'DriveHandler',
    'EnvConfigErrorHandler',
    'FolderHandler',
    'SetupHandler',
    'StatusChecker',
    'with_error_handling'
]
