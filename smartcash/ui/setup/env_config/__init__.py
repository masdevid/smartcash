"""
File: smartcash/ui/setup/env_config/__init__.py
Environment configuration module for SmartCash UI setup.
"""

# Import only the main component and initializer
from .components.env_config_component import create_env_config_component
from .env_config_initializer import initialize_environment_config_ui

# Main exports
__all__ = [
    'create_env_config_component',
    'initialize_environment_config_ui'
]