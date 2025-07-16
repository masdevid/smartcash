"""
File: smartcash/ui/setup/dependency/__init__.py
Description: Simplified dependency module exports using UIModule pattern.
"""

# Import simplified UIModule functions
from .dependency_uimodule import (
    DependencyUIModule,
    create_dependency_uimodule,
    get_dependency_uimodule,
    reset_dependency_uimodule,
    initialize_dependency_ui
)

__all__ = [
    # UIModule pattern (simplified implementation)
    "DependencyUIModule",
    "create_dependency_uimodule", 
    "get_dependency_uimodule",
    "reset_dependency_uimodule",
    "initialize_dependency_ui"
]