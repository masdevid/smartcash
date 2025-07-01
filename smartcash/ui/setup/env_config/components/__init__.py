"""
File: smartcash/ui/setup/env_config/components/__init__.py

UI Components for Environment Configuration.

This package provides reusable UI components for the environment configuration
module, following a consistent design pattern and style guide.

Components:
    - create_env_info_panel: Displays environment information
    - create_setup_summary: Shows setup progress and results
    - create_tips_requirements: Provides helpful tips and requirements
"""

from .env_info_panel import create_env_info_panel
from .setup_summary import create_setup_summary
from .tips_panel import create_tips_requirements

__all__ = [
    # Component factory functions
    'create_env_info_panel',
    'create_setup_summary',
    'create_tips_requirements',
]
