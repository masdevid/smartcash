"""
File: smartcash/ui/setup/dependency/__init__.py
Dependency Management Module.

This module provides dependency management functionality for SmartCash.
"""

# Import main initialization function
from .dependency_initializer import (
    DependencyInitializer,
    initialize_dependency_ui
)

# Alias for backward compatibility
init_dependency = initialize_dependency_ui

__all__ = [
    # Classes
    'DependencyInitializer',
    'initialize_dependency_ui'
]