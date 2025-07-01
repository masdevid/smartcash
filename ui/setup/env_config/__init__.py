"""
File: smartcash/ui/setup/env_config/__init__.py
Deskripsi: Entry point untuk environment configuration module SmartCash
"""

# Import main exports for clean API
from smartcash.ui.setup.env_config.env_config_initializer import (
    initialize_env_config_ui,
    EnvConfigInitializer
)

# Main exports
__all__ = [
    'initialize_env_config_ui',
    'EnvConfigInitializer',
]