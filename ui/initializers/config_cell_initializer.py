"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Config cell initializer with shared state and YAML persistence

This module provides the ConfigCellInitializer class which serves as the orchestration layer
for configuration UIs. It handles initialization, lifecycle management, and component
registration while delegating UI component creation to the components module.
"""

from __future__ import annotations

# Standard library
import logging
from abc import ABC, abstractmethod
from ast import Return
from typing import Any, Dict, Generic, Optional, TypeVar, Union, List
# Third-party
import ipywidgets as widgets

# SmartCash - UI Components
from smartcash.ui.config_cell.components import component_registry
from smartcash.ui.config_cell.components.ui_parent_components import ParentComponentManager, create_parent_component
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.handlers.error_handler import create_error_response
from smartcash.ui.utils.logger_bridge import UILoggerBridge
from smartcash.ui.utils.logging_utils import (
    restore_stdout
)

# Type variables
T = TypeVar('T', bound=ConfigCellHandler)

# Logger setup
logger = logging.getLogger(__name__)

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
        config: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        component_id: Optional[str] = None,
        logger_bridge: Optional[UILoggerBridge] = None,
        title: Optional[str] = None,
        children: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Initialize the config cell with optional config and parent ID.
        
        Args:
            config: Configuration dictionary
            parent_id: Optional parent component ID for hierarchical relationships
            component_id: Optional unique identifier for this component
            logger_bridge: Optional logger bridge for UI logging
            title: Optional title for the component
            children: Optional list of child component configurations
            **kwargs: Additional configuration parameters
        """
        self.config = config or {}
        self.parent_id = parent_id
        self.component_id = component_id or self.__class__.__name__
        self.title = title or self.component_id
        
        # Initialize UI components dictionary
        self.ui_components: Dict[str, Any] = {}
        
        # Store the logger bridge if provided, will be initialized after UI components are created
        self._logger_bridge = logger_bridge
        self._handler: Optional[T] = None  # Initialize the protected attribute
        self._is_initialized = False
        self._suppress_output = False
        self._original_stdout = None
        self._original_stderr = None
        self._logger = logger.getChild(self.component_id)
        
        # Initialize parent component manager
        self.parent_component = ParentComponentManager(
            parent_id=self.component_id,
            title=self.title
        )
        
        # Initialize logger bridge after parent component is created
        if self._logger_bridge is None:
            self._logger_bridge = UILoggerBridge(
                ui_components={
                    'parent': self.parent_component,
                    'container': getattr(self.parent_component, 'container', None)
                },
                logger_name=f"{self.__class__.__name__.lower()}_bridge"
            )
        
        # Store children configurations for lazy initialization
        self._children_config = children or []
        self._children: List[Any] = []
    
    def _setup_output_suppression(self) -> None:
        """Set up output suppression using the project's logging utilities.
        
        This method uses the centralized logging utilities to suppress output
        during initialization, preventing cluttering the notebook with unnecessary messages.
        """
        from smartcash.ui.utils.logging_utils import (
            setup_aggressive_log_suppression,
            setup_stdout_suppression
        )
        
        # Set up aggressive log suppression for known noisy libraries
        setup_aggressive_log_suppression()
        
        # Set up stdout/stderr suppression if we have UI components
        if hasattr(self, 'ui_components') and self.ui_components:
            setup_stdout_suppression()
            
        self._logger.debug("Output suppression enabled")
        
    def _restore_output(self) -> None:
        """Restore the original output settings using the project's logging utilities."""
        from smartcash.ui.utils.logging_utils import (
            restore_stdout,
            allow_tqdm_display
        )
        
        # Restore stdout/stderr if they were suppressed
        if hasattr(self, 'ui_components') and self.ui_components:
            restore_stdout()
            
        # Ensure tqdm is allowed to display progress bars
        allow_tqdm_display()
        
        self._logger.debug("Output settings restored")

    def _setup_logging(self) -> None:
        """Initialize logging infrastructure and redirect all logs to parent's log accordion."""
        try:
            # Initialize logger bridge with parent's UI components if available
            parent_components = {}
            if self.parent_id:
                from smartcash.ui.config_cell.components.component_registry import component_registry
                parent = component_registry.get_component(self.parent_id)
                if parent and hasattr(parent, 'get'):
                    parent_components = parent
            
            # Use parent's UI components if available, otherwise use our own
            ui_components = parent_components.get('ui_components', {}) or self.ui_components
            
            # Initialize logger bridge
            self._logger_bridge = UILoggerBridge(
                ui_components=ui_components,
                logger_name=f"smartcash.ui.{self.component_id}"
            )
            
            # Set up the logger
            self._logger = self._logger_bridge.logger
            
            # Mark UI as ready to flush any buffered logs
            if hasattr(self._logger_bridge, 'set_ui_ready'):
                self._logger_bridge.set_ui_ready(True)
                
            self._logger.debug(f"Logging initialized for {self.component_id}")
            
        except Exception as e:
            # Fallback to basic logging if UI logging setup fails
            import logging
            self._logger = logging.getLogger(f"smartcash.ui.{self.component_id}")
            self._logger.warning(f"Failed to initialize UI logging: {str(e)}", exc_info=True)
    
    def _setup_component_registry(self) -> None:
        """Register this component in the component registry.
        
        Registers the component with its full module path and sets up parent-child
        relationships if a parent_id is specified.
        """
        # Create component ID using module hierarchy (e.g., 'dataset.split')
        self._component_id = (
            f"{self.parent_id}.{self.component_id}" 
            if self.parent_id 
            else self.component_id
        )
        
        # Only register if not already registered
        if not component_registry.get_component(self._component_id):
            component_registry.register_component(
                component_id=self._component_id,
                component={
                    **self.ui_components,
                    'container': self.parent_component.container,
                    'content_area': self.parent_component.content_area
                },
                parent_id=self.parent_id
            )
            
            # Set up children if any
            if self._children_config:
                self._initialize_children()
    
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
        
    @abstractmethod
    def setup_handlers(self) -> None:
        """Set up event handlers for the UI components.
        
        This method should be implemented by subclasses to set up any
        event handlers, observers, or callbacks needed for the UI to function.
        The parent class's implementation should be called first using super().
        """
        pass
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> 'widgets.Widget':
        """Initialize the UI components and return the root widget.
        
        Args:
            config: Optional configuration to override the initial config
            
        Returns:
            The root widget containing the initialized UI
        """
        if config is not None:
            self.config = config
            
        try:
            # Set up output suppression if needed
            if self._suppress_output:
                self._setup_output_suppression()
            
            # Create the handler
            self.handler = self.create_handler()
            
            # Create UI components using the parent component system
            self._setup_ui_components()
            
            # Initialize child components if any
            self._initialize_children()
            
            # Set up event handlers
            self.setup_handlers()
            
            # Register the component
            self._register_component()
            
            # Mark as initialized
            self._is_initialized = True
            
            # Return the container widget from parent component
            return self.parent_component.container
            
        except Exception as e:
            error_msg = f"Failed to initialize {self.__class__.__name__}: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            return create_error_response(error_msg, str(e))
            
        finally:
            # Restore output settings if they were suppressed
            if self._suppress_output:
                self._restore_output()
                
    def _setup_ui_components(self) -> None:
        """Set up UI components using the parent component system."""
        # Create the main UI components
        self.ui_components = self.create_ui_components(self.config)
        
        # Add main components to the parent component
        if 'container' in self.ui_components:
            # If the component provides its own container, use it as the main content
            self.parent_component.content_area.children = (self.ui_components['container'],)
        else:
            # Otherwise, add all components to the content area
            widgets_to_add = [
                widget for key, widget in self.ui_components.items()
                if isinstance(widget, widgets.Widget)
            ]
            self.parent_component.content_area.children = tuple(widgets_to_add)
    
    def _initialize_children(self) -> None:
        """Initialize child components if any are configured."""
        if not self._children_config:
            return
            
        for child_config in self._children_config:
            try:
                # Create child component using the factory function
                child = create_parent_component(
                    parent_id=f"{self.component_id}.{child_config['id']}",
                    **{k: v for k, v in child_config.items() if k != 'id'}
                )
                self._children.append(child)
                
                # Add child to the parent component
                self.parent_component.add_child_component(
                    child_id=child_config['id'],
                    component=child,
                    config=child_config.get('config', {})
                )
                
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize child component {child_config.get('id')}: {str(e)}",
                    exc_info=True
                )
    
    def _register_component(self) -> None:
        """Register this component with the component registry and set up parent-child relationships.
        
        This method handles:
        1. Registering the component with a unique ID
        2. Setting up parent-child relationships in the UI hierarchy
        3. Registering all child components recursively
        """
        # Generate the full component ID with parent prefix if parent exists
        full_component_id = f"{self.parent_id}.{self.component_id}" if self.parent_id else self.component_id
        
        # Prepare component data with container and content area
        component_data = {
            **getattr(self, 'ui_components', {}),
            'container': getattr(self, 'container', None),
            'content_area': getattr(self, 'content_area', None)
        }
        
        # Register the main component
        component_registry.register_component(
            component_id=full_component_id,
            component=component_data,
            parent_id=self.parent_id
        )
        
        # If this is a child component (has a parent_id), add to parent's content area
        if self.parent_id:
            # Get the parent component from the registry
            parent_component = component_registry.get_component(self.parent_id)
            
            if parent_component and 'content_area' in parent_component:
                parent_content = parent_component['content_area']
                
                if parent_content and hasattr(parent_content, 'children'):
                    current_children = list(parent_content.children)
                    
                    # Add the current component's container to the parent's content area
                    if hasattr(self, 'container') and self.container and self.container not in current_children:
                        parent_content.children = tuple(current_children + [self.container])
        
        # Register all children components
        for child in getattr(self, '_children', []):
            if not (hasattr(child, 'component_id') and hasattr(child, 'parent_component')):
                continue
                
            child_parent_id = full_component_id
            child_component_id = f"{child_parent_id}.{child.component_id}"
            
            # Ensure child has ui_components attribute
            if not hasattr(child, 'ui_components'):
                child.ui_components = {}
                
            # Create component data for the child
            child_component = {
                **child.ui_components,
                'container': getattr(child, 'container', None),
                'content_area': getattr(child, 'content_area', None)
            }
            
            # Register the child component
            component_registry.register_component(
                component_id=child_component_id,
                component=child_component,
                parent_id=child_parent_id
            )
            
            # Add child's container to the current component's content area
            if (hasattr(self, 'content_area') and 
                hasattr(child, 'container') and 
                child.container is not None):
                
                current_children = list(self.content_area.children)
                if child.container not in current_children:
                    self.content_area.children = tuple(current_children + [child.container])
            
            # Recursively register any children of this child
            if hasattr(child, '_register_component'):
                child._register_component()
    
    def get_container(self) -> 'widgets.Widget':
        """Get the root container widget.
        
        Returns:
            The root container widget from the parent component
        """
        # Register the component before returning the container
        self._register_component()
        return self.parent_component.container

@property
def handler(self) -> T:
    """Lazy initialization of the configuration handler."""
    if self._handler is None:
        self._handler = self.create_handler()
    return self._handler

@abstractmethod
def create_handler(self) -> T:
    """Create and return a configuration handler instance.
    
    This method must be implemented by subclasses to create and return
    an instance of the appropriate configuration handler.
    
    Returns:
        An instance of a ConfigCellHandler subclass.
    """
    pass

def cleanup(self) -> None:
    """Release all resources and unregister components."""
    try:
        self._logger.debug(f"Cleaning up {self.component_id} resources")
        
        # Clean up logger bridge if it exists
        if hasattr(self, '_logger_bridge'):
            self._logger_bridge.cleanup()
        
        # Unregister components from the registry
        if hasattr(self, '_component_id'):
            component_registry.unregister_component(self._component_id)
            
            # Unregister all child components
            for child in self._children:
                if hasattr(child, 'component_id'):
                    component_registry.unregister_component(child.component_id)
        
        # Clean up handler if it exists
        if hasattr(self, '_handler') and hasattr(self._handler, 'cleanup'):
            self._handler.cleanup()
            
        self._logger.info(f"Cleaned up resources for {self.component_id}")
        
    except Exception as e:
        self._logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
    finally:
        restore_stdout()
