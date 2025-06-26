"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Config cell initializer with shared state and YAML persistence

This module provides the ConfigCellInitializer class which serves as the orchestration layer
for configuration UIs. It handles initialization, lifecycle management, and component
registration while delegating UI component creation to the components module.
"""

from __future__ import annotations

# Standard library
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

# Third-party
import ipywidgets as widgets

# SmartCash - Core
from smartcash.common.logger import get_logger

# SmartCash - UI Components
from smartcash.ui.config_cell.components import component_registry
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.handlers.error_handler import create_error_response
from smartcash.ui.utils.logger_bridge import UILoggerBridge
from smartcash.ui.utils.logging_utils import (
    restore_stdout,
    setup_aggressive_log_suppression
)

# Type variables
T = TypeVar('T', bound=ConfigCellHandler)

# Logger setup
logger = get_logger(__name__)

class ConfigCellInitializer(Generic[T], ABC):
    """Orchestrates the initialization and lifecycle of configuration cells.
    
    This abstract base class handles the core initialization flow, component registration,
    and lifecycle management of configuration UIs. It delegates UI component creation
    to the components module and focuses on orchestration.
    
    Type Parameters:
        T: Type of the configuration handler, must be a subclass of ConfigCellHandler
        
    Subclasses must implement:
        - create_handler(): Create and return a configuration handler instance
        - create_ui_components(): Create and return UI components dictionary
    """
    
    def __init__(
        self, 
        module_name: str, 
        config_filename: str, 
        parent_module: Optional[str] = None,
        is_container: bool = False,
        **kwargs
    ):
        """Initialize the configuration cell.
        
        Args:
            module_name: Unique identifier for this module (e.g., 'split', 'strategy').
            config_filename: Base filename for configuration persistence.
            parent_module: Optional parent module path (e.g., 'dataset' for 'dataset.split').
            is_container: If True, this component can contain other child components.
            **kwargs: Additional keyword arguments for future extension.
            
        Note:
            - The module hierarchy is separate from UI component hierarchy.
            - Use parent_module to define module relationships (e.g., 'dataset.split').
            - Use is_container to indicate if this component can contain other components.
        """
        self.module_name = module_name
        self.config_filename = config_filename
        self.parent_module = parent_module
        self.is_container = is_container
        
        # Setup logging and component registry
        self._setup_logging()
        self._setup_component_registry()
        
        # Initialize handler lazily
        self._handler: Optional[T] = None
    
    def _setup_logging(self) -> None:
        """Initialize logging infrastructure."""
        self.ui_components = {}
        self._logger_bridge = UILoggerBridge(
            self.ui_components, 
            f"smartcash.ui.{self.module_name}"
        )
        self.logger = self._logger_bridge.logger
        setup_aggressive_log_suppression()
    
    def _setup_component_registry(self) -> None:
        """Register this component in the component registry.
        
        Registers the component with its full module path and sets up parent-child
        relationships if a parent_module is specified.
        """
        # Create component ID using module hierarchy (e.g., 'dataset.split')
        self._component_id = (
            f"{self.parent_module}.{self.module_name}" 
            if self.parent_module 
            else self.module_name
        )
        
        # Register with component registry
        component_registry.register_component(
            component_id=self._component_id,
            component=self.ui_components,
            parent_id=self.parent_module
        )
        
        # Set up container if needed
        if self.is_container:
            self._initialize_as_container()
    
    @property
    def handler(self) -> T:
        """Lazy initialization of the configuration handler."""
        if self._handler is None:
            self._handler = self.create_handler()
        return self._handler
    
    @abstractmethod
    def create_handler(self) -> T:
        """Create and return a configuration handler instance.
        
        Returns:
            An instance of a ConfigCellHandler subclass.
        """
        pass
        
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return UI components based on the provided config.
        
        Args:
            config: Current configuration values
            
        Returns:
            Dictionary of UI components with at least a 'container' widget.
        """
        pass
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the configuration cell UI with the given config.
        
        This method orchestrates the entire initialization process:
        1. Creates or retrieves the configuration handler
        2. Loads or validates the provided configuration
        3. Delegates UI component creation to the components module
        4. Sets up event handlers and callbacks
        5. Returns the UI components dictionary
        
        Args:
            config: Optional initial configuration. If not provided, the handler's
                   load_config() method will be used to load the configuration.
                   
        Returns:
            Dictionary of UI components that can be displayed or further customized.
            The dictionary will always contain a 'container' widget as the root element.
            
        Raises:
            RuntimeError: If initialization fails due to configuration errors
        """
        try:
            # Update handler with provided config
            if config is not None:
                self.handler.update(config)
            
            # Delegate UI creation to components module
            ui_components = self.create_ui_components(self.handler.config)
            self.ui_components.update(ui_components)
            
            # Ensure container exists
            if 'container' not in self.ui_components:
                self.ui_components['container'] = widgets.VBox()
            
            self.logger.info(f"Successfully initialized {self.module_name} UI")
            return self.ui_components
            
        except Exception as e:
            error_msg = f"Failed to initialize {self.module_name} UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_ui(error_msg)
        
    def connect_to_parent(self) -> None:
        """Connect this component to its parent in the component hierarchy."""
        if not self.parent_module:
            self.logger.debug("No parent module specified, skipping parent connection")
            return
            
        parent_id = f"{self.parent_module}.parent"
        parent = component_registry.get_component(parent_id)
        
        if not parent:
            error_msg = f"Parent component {parent_id} not found in registry"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            # Add this component to the parent's content area
            if hasattr(parent, 'add_child'):
                parent.add_child(self.ui_components['container'])
                self.logger.debug(f"Connected {self._component_id} to parent {parent_id}")
            else:
                self.logger.warning(f"Parent {parent_id} does not support add_child")
                
        except Exception as e:
            error_msg = f"Failed to connect to parent {parent_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def setup_handlers(self) -> None:
        """Set up event handlers and callbacks for UI components.
        
        This method is called after UI components are created and can be overridden
        to set up any additional event handlers, observers, or callbacks needed
        for the UI to function properly.
        
        The default implementation sets up cleanup handlers for when the UI is closed
        or refreshed.
        
        Note:
            - This method is called automatically during initialization.
            - Access UI components through self.ui_components.
            - Connect widgets to handler methods as needed.
            - Override this method to add custom handlers, but make sure to call super().setup_handlers()
            
        Example:
            def setup_handlers(self):
                # Call parent implementation
                super().setup_handlers()
                
                # Connect a button click to a handler method
                self.ui_components['save_button'].on_click(self._on_save_clicked)
                
                # Set up an observer on a text input
                self.ui_components['text_input'].observe(
                    self._on_text_changed,
                    names='value'
                )
        """
        try:
            # Register cleanup function for when the cell is re-executed
            self._register_cleanup()
            
            # Log that handlers have been set up
            self.logger.debug(f"UI event handlers set up for {self.module_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to set up UI handlers: {str(e)}", exc_info=True)
    
    def _register_cleanup(self) -> None:
        """Register cleanup function for when the cell is re-executed.
        
        This ensures that resources are properly cleaned up when the UI is refreshed.
        """
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            
            if ipython is not None:
                # Register cleanup function to run before cell execution
                def cleanup():
                    try:
                        self.logger.debug(f"Cleaning up {self.module_name} resources")
                        
                        # Clean up logger bridge if it exists
                        if hasattr(self, '_logger_bridge') and self._logger_bridge:
                            if hasattr(self._logger_bridge, 'cleanup'):
                                self._logger_bridge.cleanup()
                            
                        # Restore stdout/stderr
                        if hasattr(sys, '_original_stdout_saved'):
                            sys.stdout = sys._original_stdout_saved
                            del sys._original_stdout_saved
                        if hasattr(sys, '_original_stderr_saved'):
                            sys.stderr = sys._original_stderr_saved
                            del sys._original_stderr_saved
                            
                    except Exception as e:
                        # Use print as logging might not be available during cleanup
                        print(f"Error during cleanup: {str(e)}")
                
                # Register the cleanup function
                ipython.events.register('pre_run_cell', lambda: cleanup())
                
        except Exception as e:
            self.logger.warning(f"Failed to register cleanup function: {str(e)}", exc_info=True)
    
    def _initialize_logger_bridge(self) -> None:
        """Initialize the logger bridge for UI logging.
        
        This sets up a bridge between the Python logging system and the UI,
        allowing log messages to be displayed in the application's log panel.
        
        The logger bridge is already initialized in __init__, this method
        configures it with the current UI components.
        """
        if not hasattr(self, '_logger_bridge') or not self._logger_bridge:
            self.logger.warning("Logger bridge not initialized, creating a new one")
            self._logger_bridge = UILoggerBridge(
                self.ui_components,
                f"smartcash.ui.{self.module_name}"
            )
        
        try:
            # Update the logger instance to use the bridge
            self.logger = self._logger_bridge.logger
            
            # Log a test message to verify logging is working
            self.logger.debug(f"Logger bridge initialized for {self.module_name}")
            
        except Exception as e:
            # Fallback to basic logging if bridge initialization fails
            self.logger = get_logger(f"smartcash.ui.{self.module_name}")
            self.logger.warning(f"Failed to configure logger bridge: {str(e)}", exc_info=True)
            
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
            
            # Create UI components using current config
            self.ui_components = self.create_ui_components(self.handler.config)
            
            # Update logger bridge with the new UI components
            self._logger_bridge.ui_components = self.ui_components
            
            # Initialize logger bridge with the UI components
            self._initialize_logger_bridge()
            
            # Mark UI as ready to flush any buffered logs
            self._logger_bridge.set_ui_ready(True)
            
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
            error_msg = f"Failed to initialize {self.module_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_ui(error_msg)
            
        finally:
            restore_stdout()
    
    def _initialize_as_container(self) -> None:
        """Initialize this component as a container for other components.
        
        Creates and registers a container widget that can hold child components.
        The container is registered with a '.container' suffix in the component registry.
        """
        from smartcash.ui.config_cell.components.ui_factory import create_container
        
        self.logger.debug(f"Initializing container for {self._component_id}")
        
        # Create container using the factory
        container_ui = create_container(
            title=f"{self.module_name} Configuration",
            container_id=self._component_id
        )
        
        # Update UI components with container
        self.ui_components.update(container_ui)
        
        # Register container in the registry
        container_id = f"{self._component_id}.container"
        component_registry.register_component(
            component_id=container_id,
            component=container_ui,
            parent_id=self._component_id
        )
        
        self.logger.info(f"Initialized container {container_id}")
    
    def cleanup(self) -> None:
        """Release all resources and unregister components."""
        try:
            self.logger.debug(f"Cleaning up {self.module_name} resources")
            
            # Clean up logger bridge if it exists
            if hasattr(self, '_logger_bridge'):
                self._logger_bridge.cleanup()
            
            # Unregister components from the registry
            if hasattr(self, '_component_id'):
                component_registry.unregister_component(self._component_id)
                if self.is_parent:
                    component_registry.unregister_component(f"{self._component_id}.parent")
            
            # Clean up handler if it exists
            if hasattr(self, '_handler') and hasattr(self._handler, 'cleanup'):
                self._handler.cleanup()
                
            self.logger.info(f"Cleaned up resources for {self.module_name}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        finally:
            restore_stdout()
    
    def _create_error_ui(
        self, 
        error_message: str, 
        details: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a fallback UI for error conditions."""
        error_title = title or f"Error in {self.module_name}"
        return create_error_response(
            error_message=error_message,
            title=error_title,
            include_traceback=bool(details)
        )
