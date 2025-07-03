"""
File: smartcash/ui/setup/env_config/configs/__init__.py

Configuration module untuk environment setup.
"""

from smartcash.ui.setup.env_config.configs.defaults import DEFAULT_CONFIG
from smartcash.ui.setup.env_config.configs.validator import validate_config, get_validation_errors

__all__ = [
    'DEFAULT_CONFIG',
    'validate_config',
    'get_validation_errors'
]