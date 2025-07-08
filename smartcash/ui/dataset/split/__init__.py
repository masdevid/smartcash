"""
Dataset Split Module.

This module provides functionality for splitting datasets into training, validation, and test sets.
"""

from typing import Dict, Any, Optional

from .split_initializer import (
    create_split_config_cell,
    get_split_config_components,
    SplitInitializer
)
from .handlers.config_handler import SplitConfigHandler

# Backward compatibility aliases
create_split_config_ui = create_split_config_cell
display_split_config = create_split_config_cell
create_split_init = create_split_config_cell
create_split_ui = create_split_config_cell
display_split_ui = create_split_config_cell
get_split_config = get_split_config_components

# Public API
def initialize_split_ui(config: Optional[Dict[str, Any]] = None) -> None:
    """🎯 Initialize the dataset split UI with optional configuration.
    
    Args:
        config: Optional configuration dictionary
    """
    return create_split_config_cell(config)


def get_split_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """📦 Get dataset split UI components for programmatic access.
    
    Returns a dictionary of UI components for programmatic manipulation
    without displaying the UI in the notebook.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing UI components
    """
    return get_split_config_components(config)


# Export public API
__all__ = [
    # Main API
    'create_split_config_cell',
    'get_split_config_components',
    'SplitInitializer',
    'SplitConfigHandler',
    
    # Convenience functions
    'initialize_split_ui',
    'get_split_ui_components',
    'get_split_config',
    
    # Backward compatibility
    'create_split_config_ui',
    'display_split_config',
    'create_split_init',
    'create_split_ui',
    'display_split_ui'
]
