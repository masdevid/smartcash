"""
Dependency Management UI Components.

This module provides the main UI components for dependency management.
"""

from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
from smartcash.ui.setup.dependency.components.summary_panel import DependencySummaryPanel
from smartcash.ui.setup.dependency.components.ui_categories_section import create_categories_section
from smartcash.ui.setup.dependency.components.ui_custom_packages_section import create_custom_packages_section

__all__ = [
    'create_dependency_main_ui',
    'DependencySummaryPanel',
    'create_categories_section',
    'create_custom_packages_section'
]