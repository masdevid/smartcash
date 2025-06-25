"""
Dependency installer utilities.

This module provides a collection of utilities for managing Python package dependencies,
including installation, status checking, and UI components.
"""

# Standard library imports
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Import and expose the main initialization function
from smartcash.ui.setup.dependency.dependency_initializer import initialize_dependency_ui

# Re-export core utilities
from smartcash.ui.setup.dependency.utils.core import *
from smartcash.ui.setup.dependency.utils.package import *
from smartcash.ui.setup.dependency.utils.system import *
from smartcash.ui.setup.dependency.utils.reporting import *

# Lazy imports for UI components to prevent circular imports
if TYPE_CHECKING:
    from smartcash.ui.setup.dependency.utils.ui import *

__all__ = [
    'initialize_dependency_ui',
    # Other exports from core, package, system, and reporting modules
]

# Import modules to get their __all__ attributes
from smartcash.ui.setup.dependency.utils import core, package, system, reporting

# Re-export all symbols from submodules
__all__ = []
__all__.extend(getattr(core, '__all__', []))
__all__.extend(getattr(package, '__all__', []))
__all__.extend(getattr(system, '__all__', []))
__all__.extend(getattr(reporting, '__all__', []))

# Lazy loading for UI components
def __getattr__(name: str) -> Any:
    """Lazy load UI components to prevent circular imports."""
    if name in {'get_selected_packages', 'reset_package_selections'}:
        from smartcash.ui.setup.dependency.utils.ui.utils import (
            get_selected_packages,
            reset_package_selections
        )
        globals()[name] = locals()[name]
        return globals()[name]
    
    if name in {'with_button_state', 'create_button_state_handler'}:
        from smartcash.ui.setup.dependency.utils.ui.components.buttons import (
            with_button_state,
            create_button_state_handler
        )
        globals()[name] = locals()[name]
        return globals()[name]
    
    if name in {'ProgressSteps', 'create_operation_context', 'update_status_panel'}:
        from smartcash.ui.setup.dependency.utils.ui.state import (
            ProgressSteps,
            create_operation_context,
            update_status_panel
        )
        globals()[name] = locals()[name]
        return globals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Add version information
__version__ = "1.0.0"