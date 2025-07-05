"""Base UI component class with common functionality."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
import ipywidgets as widgets

if TYPE_CHECKING:
    from logging import Logger

class BaseUIComponent(ABC):
    """Base class for all UI components with common functionality."""
    
    def __init__(self, component_name: str, logger: Optional['Logger'] = None):
        """Initialize base component.
        
        Args:
            component_name: Unique name for this component instance
            logger: Optional logger instance. If not provided, a default logger will be created.
        """
        self.component_name = component_name
        self._ui_components: Dict[str, Any] = {}
        self._initialized = False
        
        # Initialize logger
        self.logger = logger or logging.getLogger(f"smartcash.ui.{self.__class__.__name__}")
        self.logger.debug(f"Initializing {self.__class__.__name__} with component_name: {component_name}")
        
    def initialize(self) -> None:
        """Initialize the component and its UI elements."""
        if not self._initialized:
            self._create_ui_components()
            self._setup_event_handlers()
            self._initialized = True
            
    @abstractmethod
    def _create_ui_components(self) -> None:
        """Create and initialize all UI components. Must be implemented by subclasses."""
        pass
        
    def _setup_event_handlers(self) -> None:
        """Set up event handlers. Can be overridden by subclasses."""
        pass
        
    def get_component(self, name: str) -> Any:
        """Get a UI component by name.
        
        Args:
            name: Name of the component to retrieve
            
        Returns:
            The requested component or None if not found
        """
        return self._ui_components.get(name)
        
    def show(self) -> widgets.Widget:
        """Get the main widget for display.
        
        Returns:
            The main widget that can be displayed
        """
        self.initialize()
        return self._ui_components.get('container')
        
    def dispose(self) -> None:
        """Clean up resources. Should be called when the component is no longer needed."""
        self._ui_components.clear()
        self._initialized = False
