"""
File: smartcash/ui/setup/env_config/__init__.py
Deskripsi: Entry point untuk environment configuration module SmartCash
"""

# Import only main exports untuk clean API
from smartcash.ui.setup.env_config.env_config_initializer import initialize_environment_config_ui

# Main exports
__all__ = [
    'initialize_environment_config_ui',
]