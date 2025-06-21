"""
File: smartcash/ui/setup/env_config/__init__.py
Deskripsi: Init file dengan clean exports dan constants integration
"""

from .components.env_config_component import create_env_config_component
from .env_config_initializer import initialize_env_config_ui, initialize_environment_config_ui
from .constants import (
    REQUIRED_FOLDERS, CONFIG_TEMPLATES, ESSENTIAL_CONFIGS,
    PROGRESS_RANGES, STATUS_MESSAGES, PROGRESS_MESSAGES
)
from .utils import (
    update_progress_safe, hide_progress_safe, show_progress_safe,
    complete_progress_safe, error_progress_safe, reset_progress_safe,
    is_colab_environment, test_drive_readiness, validate_setup_integrity
)

# Main exports
__all__ = [
    # Component exports
    'create_env_config_component',
    'initialize_env_config_ui',
    'initialize_environment_config_ui',
    
    # Constants
    'REQUIRED_FOLDERS',
    'CONFIG_TEMPLATES',
    'ESSENTIAL_CONFIGS',
    'PROGRESS_RANGES',
    'STATUS_MESSAGES',
    'PROGRESS_MESSAGES',
    
    # Utils
    'update_progress_safe',
    'hide_progress_safe',
    'show_progress_safe',
    'complete_progress_safe',
    'error_progress_safe',
    'reset_progress_safe',
    'is_colab_environment',
    'test_drive_readiness',
    'validate_setup_integrity'
]

# Version info
__version__ = '1.0.0'
__author__ = 'SmartCash Team'