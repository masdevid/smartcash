"""
File: smartcash/ui/setup/dependency/__init__.py
Description: Simplified dependency module exports using UIModule pattern.
"""

# Import new UIModule functions (preferred approach)
from .dependency_uimodule import (
    DependencyUIModule,
    create_dependency_uimodule,
    get_dependency_uimodule,
    reset_dependency_uimodule,
    initialize_dependency_ui,
    display_dependency_ui,
    get_dependency_components
)

__all__ = [
    # UIModule pattern (current implementation)
    "DependencyUIModule",
    "create_dependency_uimodule", 
    "get_dependency_uimodule",
    "reset_dependency_uimodule",
    "initialize_dependency_ui",
    "display_dependency_ui",
    "get_dependency_components"
]