"""
File: smartcash/ui/setup/env_config/__init__.py

Environment Configuration Module - Refactored.

Module ini menyediakan interface utama untuk inisialisasi dan manajemen
environment configuration UI components dalam SmartCash.

Exports:
    - EnvConfigInitializer: Class utama untuk environment configuration initialization
    - initialize_env_config_ui: Function untuk quick UI initialization
"""

# Import main exports untuk clean API
from smartcash.ui.setup.env_config.env_config_initializer import (
    initialize_env_config_ui,
    EnvConfigInitializer
)

# Main exports
__all__ = [
    'initialize_env_config_ui',
    'EnvConfigInitializer',
]