"""
File: smartcash/ui/setup/env_config/handlers/__init__.py

Handlers for Environment Configuration.

This package provides handler classes that manage different aspects of the 
environment configuration setup, following the CommonInitializer pattern.

Handlers:
    - ConfigHandler: Manages configuration file operations
    - DriveHandler: Handles Google Drive integration
    - EnvConfigErrorHandler: Centralized error handling
    - FolderHandler: Manages directory structure
    - SetupHandler: Orchestrates the setup workflow
    - StatusChecker: Verifies environment status
"""

# Import from main handlers directory
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler as BaseConfigHandler

# Import local handlers
from ..configs.config_handler import ConfigHandler
from .drive_handler import DriveHandler
from .folder_handler import FolderHandler
from .setup_handler import SetupHandler
from .status_checker import StatusChecker

__all__ = [
    # Environment configuration handlers
    'ConfigHandler',
    'DriveHandler',
    'FolderHandler',
    'SetupHandler',
    'StatusChecker',
]
