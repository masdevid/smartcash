"""
File: smartcash/ui/setup/__init__.py
Deskripsi: Modul untuk setup aplikasi SmartCash
"""

from smartcash.ui.setup.env_config import initialize_env_config_ui
from smartcash.ui.setup.dependency_installer import initialize_dependency_installer

__all__ = ['initialize_env_config_ui', 'initialize_dependency_installer']