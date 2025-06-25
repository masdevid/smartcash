"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Config cell initializer with shared state and YAML persistence

This module provides functionality to initialize and manage configuration cells
with shared state across notebook environments. It handles UI creation, state
management, and YAML-based persistence.
"""

from typing import Dict, Any, Optional, Type, Callable, List
import ipywidgets as widgets
import traceback
from pathlib import Path
import os

from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs, restore_stdout

# Import components
from smartcash.ui.config_cell.components.ui_components import create_config_cell_ui
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.utils.error_utils import create_error_fallback

# Shared state registry for config handlers
_shared_handlers: Dict[str, ConfigCellHandler] = {}

def get_shared_handler(module_name: str, parent_module: Optional[str] = None) -> ConfigCellHandler:
    """Get or create a shared config handler for the given module.
    
    Args:
        module_name: Name of the module
        parent_module: Optional parent module name for namespacing
        
    Returns:
        ConfigCellHandler: Shared handler instance for the module
    """
    key = f"{parent_module}.{module_name}" if parent_module else module_name
    if key not in _shared_handlers:
        _shared_handlers[key] = ConfigCellHandler(module_name, parent_module)
    return _shared_handlers[key]

def create_error_fallback(
    error_message: str, 
    traceback: Optional[str] = None
) -> Dict[str, Any]:
    """Create a fallback UI component to display error messages.
    
    Args:
        error_message: The main error message to display
        traceback: Optional detailed traceback information
        
    Returns:
        Dict[str, Any]: Dictionary containing the error UI component
    """
    from smartcash.ui.components import create_error_component
    return create_error_component("Config Cell Initialization Error", error_message, traceback)

def _update_status(
    ui_components: Dict[str, Any], 
    message: str, 
    status_type: str = "info"
) -> None:
    """Update the status bar with a message and apply appropriate styling.
    
    Args:
        ui_components: Dictionary containing UI components
        message: Status message to display
        status_type: Type of status ('info', 'success', 'warning', 'error')
    """
    if 'status_bar' in ui_components and hasattr(ui_components['status_bar'], 'value'):
        status_bar = ui_components['status_bar']
        status_bar.value = message
        
        # Apply color coding based on status type
        status_colors = {
            'info': 'blue',
            'success': 'green',
            'warning': 'orange',
            'error': 'red'
        }
        status_bar.style.text_color = status_colors.get(status_type.lower(), 'black')
    
    # Log the status message using the appropriate log level
    logger = get_logger(__name__)
    log_method = getattr(logger, status_type, logger.info)
    log_method(f"[{status_type.upper()}] {message}")


def create_config_cell(
    module_name: str,
    config_filename: str,
    env: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    parent_module: Optional[str] = None,
    config_handler_class: Optional[Type] = None,
    **kwargs
) -> Dict[str, Any]:
    """Legacy factory function for backward compatibility.
    
    This is maintained for backward compatibility with existing code.
    New code should use initialize_config_cell() directly.
    
    Args:
        module_name: Name of the module
        config_filename: Base name for the config file
        env: Environment configuration (unused, for backward compatibility)
        config: Initial configuration
        parent_module: Optional parent module name
        config_handler_class: Optional custom config handler class
        **kwargs: Additional keyword arguments (for future compatibility)
        
    Returns:
        Dict[str, Any]: Dictionary containing UI components and handlers
    """
    return initialize_config_cell(
        module_name=module_name,
        config_filename=config_filename,
        config_handler_class=config_handler_class,
        parent_module=parent_module,
        env=env,
        config=config
    )

def connect_config_to_parent(
    ui_components: Dict[str, Any],
    parent_components: Dict[str, Any],
    module_name: str,
    parent_module: Optional[str] = None
) -> None:
    """Connect child config UI components to a parent UI container.
    
    This function establishes the parent-child relationship between UI components,
    allowing for proper nesting and state sharing. It handles both the visual
    connection of containers and the sharing of configuration state.
    
    Args:
        ui_components: Dictionary of child UI components to connect
        parent_components: Dictionary of parent UI components
        module_name: Name of the child module (for namespacing)
        parent_module: Optional parent module name (for hierarchical configs)
        
    Example:
        ```python
        # In a parent component
        def initialize_parent_ui():
            parent_ui = {}
            # ... setup parent UI ...
            
            # Create and connect child config
            child_ui = initialize_config_cell(
                module_name="child_config",
                config_filename="child_config"
            )
            connect_config_to_parent(
                ui_components=child_ui,
                parent_components=parent_ui,
                module_name="child_config"
            )
            return parent_ui
        ```
    """
    try:
        # Connect container widgets if both parent and child have them
        if 'container' in parent_components and 'container' in ui_components:
            parent_container = parent_components['container']
            child_container = ui_components['container']
            
            # Preserve existing children and append the new one
            parent_container.children = tuple(list(parent_container.children) + [child_container])
            
            # Add a small margin between components
            if hasattr(child_container, 'layout') and hasattr(child_container.layout, 'margin'):
                child_container.layout.margin = '0 0 10px 0'
        
        # Share config handler if parent expects it
        if '_config_handler' in ui_components and 'config' in parent_components:
            parent_components['_config_handler'] = ui_components['_config_handler']
            
            # If parent has a method to handle config updates, call it
            if hasattr(parent_components, 'on_config_connected'):
                parent_components.on_config_connected(ui_components['_config_handler'])
                
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to connect config to parent: {str(e)}")
        logger.debug(traceback.format_exc())

    # Callback management with parent module support