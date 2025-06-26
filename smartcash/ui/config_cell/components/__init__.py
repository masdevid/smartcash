"""
Config Cell UI Components

This package provides the building blocks for creating and managing
configurable UI components in a hierarchical structure.
"""

from .component_registry import component_registry
from .ui_factory import (
    create_config_summary_panel,
    create_log_components,
    create_info_components,
    create_config_cell_ui
)
from .ui_parent_components import (
    ParentComponentManager,
    create_parent_component
)

# Re-export public API
__all__ = [
    # Core components
    'component_registry',
    'ParentComponentManager',
    'create_parent_component',
    
    # Factory functions
    'create_config_summary_panel',
    'create_log_components',
    'create_info_components',
    'create_config_cell_ui'
]
