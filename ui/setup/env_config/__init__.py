"""
File: smartcash/ui/setup/env_config/__init__.py

Environment Configuration Module.

This module provides the main interface for initializing and managing
the environment configuration UI components in SmartCash.

Exports:
    - EnvConfigInitializer: Main class for environment configuration initialization
    - initialize_env_config_ui: Convenience function for quick UI initialization
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