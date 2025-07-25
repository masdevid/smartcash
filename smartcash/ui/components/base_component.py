"""Base UI component class with common functionality and error handling."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING, Callable, TypeVar, Type
import logging
import ipywidgets as widgets

from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.core.errors.handlers import CoreErrorHandler
from smartcash.ui.core.decorators.error_decorators import handle_errors
from smartcash.ui.core.errors.exceptions import UIComponentError
from smartcash.ui.core.errors.context import ErrorContext
from smartcash.ui.core.errors.validators import safe_component_operation

if TYPE_CHECKING:
    from logging import Logger
    from smartcash.ui.core.errors import ErrorComponent

T = TypeVar('T')

class BaseUIComponent(ABC):
    """Base class for all UI components with common functionality."""
    
    def __init__(
        self, 
        component_name: str, 
        logger: Optional['Logger'] = None,
        error_handler: Optional[CoreErrorHandler] = None
    ):
        """Initialize base component with error handling.
        
        Args:
            component_name: Unique name for this component instance
            logger: Optional logger instance. If not provided, a default logger will be created.
            error_handler: Optional error handler instance. If not provided, a default will be created.
        """
        self.component_name = component_name
        self._ui_components: Dict[str, Any] = {}
        self._initialized = False
        
        # Initialize logger
        self.logger = logger or logging.getLogger(f"smartcash.ui.{self.__class__.__name__}")
        try:
            self.logger.debug(f"Initializing {self.__class__.__name__} with component_name: {component_name}")
        except (TypeError, AttributeError):
            # Fallback if logger is not properly initialized
            logging.getLogger(__name__).debug(f"Initializing {self.__class__.__name__} with component_name: {component_name}")
        
        # Initialize error handling
        self._error_handler = error_handler or CoreErrorHandler(
            component_name=component_name,
            logger=self.logger
        )
        
    @handle_errors(
        error_msg="Failed to initialize UI component",
        level=ErrorLevel.ERROR,
        context_attr="UI Component"
    )
    def initialize(self) -> None:
        """Initialize the component and its UI elements with error handling.
        
        Raises:
            UIComponentError: If initialization fails
        """
        if not self._initialized:
            with ErrorContext(component=self.component_name, operation="initialize"):
                self._create_ui_components()
                self._setup_event_handlers()
                self._initialized = True
            
    @abstractmethod
    def _create_ui_components(self) -> None:
        """Create and initialize all UI components. Must be implemented by subclasses.
        
        This method should be decorated with @handle_errors in subclasses if specific
        error handling is needed for component creation.
        """
        pass
        
    @handle_errors(
        error_msg="Failed to set up event handlers",
        level=ErrorLevel.WARNING,
        context_attr="UI Component"
    )
    def _setup_event_handlers(self) -> None:
        """Set up event handlers with error handling.
        
        This method can be overridden by subclasses to set up event handlers.
        Any errors during event handler setup will be logged but won't prevent
        the component from initializing.
        """
        pass
        
    @handle_errors(
        error_msg="Failed to get component",
        level=ErrorLevel.ERROR,
        context_attr="component_name"
    )
    def get_component(self, name: str) -> Any:
        """Safely get a UI component by name with error handling.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The requested UI component or None if not found
            
        Raises:
            UIComponentError: If the component exists but cannot be accessed
        """
        return self._ui_components.get(name)
        
    @handle_errors(
        error_msg="Failed to display component",
        level=ErrorLevel.ERROR,
        context_attr="UI Component"
    )
    def show(self) -> widgets.Widget:
        """Get the main widget for display with error handling.
        
        Returns:
            The main widget that can be displayed
            
        Raises:
            UIComponentError: If the component cannot be displayed
        """
        self.initialize()
        widget = self._ui_components.get('container')
        if widget is None:
            raise UIComponentError(f"No container widget found in {self.component_name}")
        return widget
        
    def dispose(self) -> None:
        """Clean up resources. Should be called when the component is no longer needed."""
        self._ui_components.clear()
        self._initialized = False
