"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Config cell initializer with shared state and YAML persistence

This module provides functionality to initialize and manage configuration cells
with shared state and YAML persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Callable, List, TypeVar, Generic
import ipywidgets as widgets
import traceback
from pathlib import Path
import os

from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs, restore_stdout
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler

# Type variable for the handler class
T = TypeVar('T', bound=ConfigCellHandler)

class ConfigCellInitializer(Generic[T], ABC):
    """Abstract base class for config cell initializers.
    
    This class provides common functionality for initializing configuration cells,
    including output suppression, error handling, and UI component management.
    """
    
    def __init__(self, module_name: str, config_filename: str, parent_module: Optional[str] = None):
        """Initialize the config cell initializer.
        
        Args:
            module_name: Name of the module
            config_filename: Base name for the config file
            parent_module: Optional parent module name for namespacing
        """
        self.module_name = module_name
        self.config_filename = config_filename
        self.parent_module = parent_module
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        self.ui_components: Dict[str, Any] = {}
        self._handler: Optional[T] = None
    
    @property
    def handler(self) -> T:
        """Get the config handler instance."""
        if self._handler is None:
            self._handler = self.create_handler()
        return self._handler
    
    @abstractmethod
    def create_handler(self) -> T:
        """Create and return a config handler instance.
        
        Returns:
            An instance of a ConfigHandler subclass
        """
        pass
    
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return UI components.
        
        Args:
            config: Current configuration
            
        Returns:
            Dictionary of UI components
        """
        pass
    
    def setup_handlers(self) -> None:
        """Setup event handlers for UI components."""
        pass
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the config cell with the given configuration.
        
        Args:
            config: Optional initial configuration
            
        Returns:
            Dictionary containing UI components and handlers
        """
        suppress_all_outputs()
        try:
            # Initialize handler with config
            if config:
                self.handler.update_config(config)
            
            # Create UI components
            self.ui_components = self.create_ui_components(self.handler.config)
            
            # Setup event handlers
            self.setup_handlers()
            
            # Add common UI elements
            if 'container' not in self.ui_components:
                self.ui_components['container'] = widgets.VBox()
            
            # Store the handler for later use
            self.ui_components['_config_handler'] = self.handler
            
            return self.ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.module_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return self._create_error_ui(str(e))
            
        finally:
            restore_stdout()
    
    def _create_error_ui(self, error_message: str) -> Dict[str, Any]:
        """Create a fallback UI component to display error messages."""
        from smartcash.ui.components import create_error_component
        return create_error_component(
            f"{self.module_name} Initialization Error",
            error_message,
            traceback.format_exc(),
            "error"
        )

# Shared state registry for config handlers (legacy support)
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
    from smartcash.ui.components.status_panel import update_status_panel
    
    # Use status_panel if available, fall back to status_bar for backward compatibility
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)
    elif 'status_bar' in ui_components and hasattr(ui_components['status_bar'], 'value'):
        # Fallback for legacy status_bar
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
    """Factory function for creating configuration cells.
    
    Args:
        module_name: Name of the module
        config_filename: Base name for the config file
        env: Environment configuration (unused, for backward compatibility)
        config: Initial configuration
        parent_module: Optional parent module name
        config_handler_class: Optional custom config handler class
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict[str, Any]: Dictionary containing UI components and handlers
    """
    logger = get_logger(__name__)
    
    # Suppress output during initialization
    suppress_all_outputs()
    
    try:
        # Get or create config handler
        if config_handler_class:
            handler = config_handler_class(config or {})
        else:
            handler = ConfigCellHandler(module_name, parent_module)
            if config:
                handler.update_config(config)
        
        # Create UI components
        ui_components = {}
        
        # Add container for the config cell
        container = widgets.VBox()
        ui_components['container'] = container
        
        # Store the handler for later use
        ui_components['_config_handler'] = handler
        
        # Add status bar
        status_bar = widgets.HTML()
        ui_components['status_bar'] = status_bar
        
        # Add the status bar to the container
        container.children = [status_bar]
        
        # Update status
        _update_status(ui_components, f"Initialized {module_name} configuration", 'info')
        
        # Restore output before returning
        restore_stdout()
        return ui_components
        
    except Exception as e:
        error_msg = f"Failed to create config cell: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        restore_stdout()  # Ensure output is restored even on error
        return create_error_fallback(
            f"Failed to initialize {module_name} configuration",
            error_msg
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