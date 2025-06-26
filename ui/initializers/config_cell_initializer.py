"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Config cell initializer with shared state and YAML persistence

This module implements a robust system for initializing and managing configuration UI cells
in Jupyter notebooks, featuring shared state management and YAML persistence.

Initialization Flow:
1. Configuration Loading:
   - Load from provided config dict if specified
   - Otherwise, load from YAML file if exists
   - Fall back to default configuration

2. UI Component Creation:
   - Create widgets based on current config
   - Set up change observers to update config
   - Initialize with current values

3. Handler Setup:
   - Create or retrieve shared handler instance
   - Register with parent if nested
   - Set up config change listeners

Shared Handler Behavior:
- Parent-Child Relationship:
  - Child configs are namespaced under parent
  - Shared state is synchronized automatically
  - Changes propagate up the hierarchy

- Orphan Configs:
  - Operate independently
  - No parent namespace
  - Self-contained state
  - Can be adopted later

Configuration Persistence:
- Automatic YAML serialization
- Per-module config files
- Hierarchical config merging
- Change tracking and validation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Callable, List, TypeVar, Generic
import ipywidgets as widgets
import traceback
from pathlib import Path
import os

from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import UILoggerBridge, create_ui_logger_bridge
from smartcash.ui.utils.logging_utils import (
    suppress_all_outputs,
    restore_stdout,
    setup_aggressive_log_suppression
)
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler

# Type variable for the handler class
T = TypeVar('T', bound=ConfigCellHandler)

class ConfigCellInitializer(Generic[T], ABC):
    """Base class for initializing configuration cells with shared state and YAML persistence.
    
    This abstract base class provides a structured approach to creating configuration UI components
    that can be embedded in Jupyter notebooks. It handles configuration loading/saving, UI state
    management, and error handling in a consistent way.
    
    Key Features:
        - Type-safe configuration handling with YAML persistence
        - Automatic UI state management
        - Built-in error handling and recovery
        - Support for nested configuration hierarchies
        - Thread-safe component management
        
    Type Parameters:
        T: Type of the configuration handler, must be a subclass of ConfigCellHandler
        
    Subclasses must implement:
        - create_handler(): Create and return a configuration handler instance
        - create_ui_components(): Create and return UI components dictionary
        
    Example:
        ```python
        class MyConfigInitializer(ConfigCellInitializer[MyConfigHandler]):
            def create_handler(self) -> MyConfigHandler:
                return MyConfigHandler()
                
            def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    'input': widgets.Text(value=config.get('value', '')),
                    'container': widgets.VBox()
                }
        ```
    """
    
    def __init__(self, module_name: str, config_filename: str, parent_module: Optional[str] = None):
        """Initialize a new configuration cell initializer.
        
        This sets up the basic infrastructure for configuration management including
        logging, state tracking, and handler initialization.
        
        Args:
            module_name: Unique identifier for this configuration module.
                        Used for logging and configuration namespacing.
            config_filename: Base filename (without extension) for configuration
                           persistence. Will be saved as '{config_filename}.yaml'.
            parent_module: Optional parent module identifier for creating nested
                         configuration hierarchies. If provided, configurations will
                         be namespaced under this parent.
                         
        Attributes:
            module_name (str): The name of this configuration module.
            config_filename (str): Base filename for configuration storage.
            parent_module (Optional[str]): Parent module identifier if nested.
            logger: Configured logger instance for this module.
            ui_components (Dict[str, Any]): Dictionary to store UI components.
            _handler (Optional[T]): Cached instance of the config handler.
        """
        self.module_name = module_name
        self.config_filename = config_filename
        self.parent_module = parent_module
        
        # Initialize with basic logger first
        self._logger_bridge = None
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        
        # Setup aggressive log suppression
        setup_aggressive_log_suppression()
        
        self.ui_components: Dict[str, Any] = {}
        self._handler: Optional[T] = None
    
    @property
    def handler(self) -> T:
        """Get the configuration handler instance, creating it if necessary.
        
        This property implements lazy initialization of the configuration handler.
        The handler is created on first access using the create_handler() method.
        
        Returns:
            T: An instance of the configuration handler.
            
        Note:
            The handler is cached after creation. To force recreation, set
            _handler to None before accessing this property.
        """
        if self._handler is None:
            self._handler = self.create_handler()
        return self._handler
    
    @abstractmethod
    def create_handler(self) -> T:
        """Create and return a new configuration handler instance.
        
        This method must be implemented by subclasses to provide a properly
        configured handler for the specific configuration type.
        
        Returns:
            T: A new instance of a ConfigCellHandler subclass.
            
        Raises:
            RuntimeError: If the handler cannot be created due to missing
                        dependencies or configuration.
                        
        Example:
            def create_handler(self) -> MyConfigHandler:
                return MyConfigHandler(
                    config_file=Path('configs') / f'{self.config_filename}.yaml',
                    default_config={
                        'setting1': 'default',
                        'setting2': 42
                    }
                )
        """
        pass
    
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return UI components for the configuration interface.
        
        This method must be implemented by subclasses to define the user interface
        for configuring the module. The UI should reflect the current configuration
        state and update the configuration when user interactions occur.
        
        Args:
            config: Current configuration dictionary containing all settings.
                   This should be used to initialize the UI component states.
                   
        Returns:
            Dict[str, Any]: A dictionary mapping string identifiers to UI components.
                          Must include at least a 'container' key with the root
                          widget that contains all other UI elements.
                          
        Note:
            - The returned dictionary is stored in the ui_components attribute.
            - All widgets should update the handler's config when changed.
            - Use ipywidgets for interactive elements.
            
        Example:
            def create_ui_components(self, config):
                input_widget = widgets.Text(
                    value=config.get('name', ''),
                    description='Name:',
                    layout={'width': '400px'}
                )
                
                def on_change(change):
                    self.handler.update_config({'name': change['new']})
                    
                input_widget.observe(on_change, names='value')
                
                return {
                    'input': input_widget,
                    'container': widgets.VBox([
                        widgets.HTML('<h3>Configuration</h3>'),
                        input_widget
                    ])
                }
        """
        pass
    
    def setup_handlers(self) -> None:
        """Set up event handlers and callbacks for UI components.
        
        This method is called after UI components are created and can be overridden
        to set up any additional event handlers, observers, or callbacks needed
        for the UI to function properly.
        
        Note:
            - This method is called automatically during initialization.
            - The default implementation does nothing.
            - Access UI components through self.ui_components.
            - Connect widgets to handler methods as needed.
            
        Example:
            def setup_handlers(self):
                # Connect a button click to a handler method
                self.ui_components['save_button'].on_click(self._on_save_clicked)
                
                # Set up an observer on a text input
                self.ui_components['text_input'].observe(
                    self._on_text_changed,
                    names='value'
                )
        """
        pass
    
    def _initialize_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Initialize the logger bridge for UI logging.
        
        This sets up a bridge between the Python logging system and the UI,
        allowing log messages to be displayed in the application's log panel.
        
        Args:
            ui_components: Dictionary containing UI components, which should include
                         a log output component if UI logging is desired.
        """
        try:
            # Create and store the logger bridge
            self._logger_bridge = create_ui_logger_bridge(
                ui_components=ui_components,
                logger_name=f"smartcash.ui.{self.module_name}"
            )
            
            # Update the logger to use the bridge
            self.logger = self._logger_bridge.logger
            
            # Mark UI as ready to flush any buffered logs
            if hasattr(self._logger_bridge, 'set_ui_ready'):
                self._logger_bridge.set_ui_ready(True)
                
            self.logger.debug(f"Logger bridge initialized for {self.module_name}")
            
        except Exception as e:
            # Fallback to basic logging if bridge initialization fails
            self.logger = get_logger(f"smartcash.ui.{self.module_name}")
            self.logger.warning(f"Failed to initialize logger bridge: {str(e)}")
            
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the configuration cell with the given configuration.
        
        This is the main entry point that sets up the configuration interface.
        It performs the following steps:
        1. Suppresses all output during initialization
        2. Updates the handler with the provided configuration
        3. Creates UI components based on the current config
        4. Sets up event handlers and callbacks
        5. Ensures a container widget exists
        6. Restores output and returns the UI components
        
        Args:
            config: Optional initial configuration dictionary. If provided,
                  this will update the current configuration before creating
                  the UI components.
                  
        Returns:
            Dict[str, Any]: A dictionary containing at least a 'container' key
                          with the root widget, along with any other UI components
                          created by create_ui_components().
                          
        Raises:
            RuntimeError: If initialization fails due to invalid configuration
                        or UI component creation errors.
                        
        Example:
            initializer = MyConfigInitializer('my_module', 'config')
            ui_components = initializer.initialize({'setting': 'value'})
            display(ui_components['container'])
        """
        suppress_all_outputs()
        try:
            # Initialize handler with config if provided
            if config:
                self.handler.update_config(config)
            
            # Create UI components using current config (without logging)
            self.ui_components = self.create_ui_components(self.handler.config)
            
            # Initialize logger bridge after UI components are created
            self._initialize_logger_bridge(self.ui_components)
            
            # Setup any additional event handlers
            self.setup_handlers()
            
            # Ensure a container widget exists
            if 'container' not in self.ui_components:
                self.ui_components['container'] = widgets.VBox()
                
            # Return both the container and components
            result = {'container': self.ui_components['container']}
            result.update(self.ui_components)
            return result
            
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
) -> None:
    """Create and immediately display a configuration cell UI.
    
    Args:
        module_name: Name of the module
        config_filename: Base name for the config file
        env: Environment configuration (unused, for backward compatibility)
        config: Initial configuration
        parent_module: Optional parent module name
        config_handler_class: Optional custom config handler class
        **kwargs: Additional keyword arguments
    """
    from IPython.display import display
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
        
        # Display the container
        display(container)
        
        # Update status
        _update_status(ui_components, f"Initialized {module_name} configuration", 'info')
        
        # Store UI components in global registry for access if needed
        global _ui_registry
        _ui_registry = ui_components
        
    except Exception as e:
        error_msg = f"Failed to create config cell: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Display error UI
        error_ui = create_error_fallback(
            f"Failed to initialize {module_name} configuration",
            error_msg
        )
        display(error_ui['widget'] if 'widget' in error_ui else error_ui)
    finally:
        restore_stdout()  # Always restore output
        
    return None  # Explicitly return None to indicate no dictionary return

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