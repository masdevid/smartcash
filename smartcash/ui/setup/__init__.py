"""
File: smartcash/ui/setup/__init__.py
Deskripsi: Setup module exports
"""

from smartcash.ui.setup.env_config import env_config_initializer
from smartcash.ui.setup.dependency import dependency_initializer

# For backward compatibility
initialize_env_config_ui = env_config_initializer.initialize_env_config_ui
initialize_dependency_ui = dependency_initializer.initialize_dependency_ui

__all__ = ['initialize_env_config_ui', 'initialize_dependency_ui']
