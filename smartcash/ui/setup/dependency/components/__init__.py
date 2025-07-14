"""
Dependency UI Components

This module contains UI component creation functions for the dependency management interface.
"""

from .dependency_ui import create_dependency_ui_components
from .package_selector import (
    get_selected_packages,
    get_custom_packages_text,
    get_all_packages,
    get_packages_by_category,
    get_package_selection_summary
)

__all__ = [
    'create_dependency_ui_components',
    'get_selected_packages',
    'get_custom_packages_text',
    'get_all_packages',
    'get_packages_by_category',
    'get_package_selection_summary'
]