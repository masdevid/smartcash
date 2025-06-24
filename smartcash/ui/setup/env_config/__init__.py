"""
File: smartcash/ui/setup/env_config/__init__.py
Deskripsi: Entry point untuk environment configuration module SmartCash
"""

# Import only main exports untuk clean API
from smartcash.ui.setup.env_config.env_config_initializer import initialize_environment_config_ui
from smartcash.ui.setup.env_config.components.env_config_component import create_env_config_component

# Main exports
__all__ = [
    'initialize_environment_config_ui',
    'create_env_config_component'
]