"""Base UI component class with common functionality."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import ipywidgets as widgets

class BaseUIComponent(ABC):
    """Base class for all UI components with common functionality."""
    
    def __init__(self, component_name: str):
        """Initialize base component.
        
        Args:
            component_name: Unique name for this component instance
        """
        self.component_name = component_name
        self._ui_components: Dict[str, Any] = {}
        self._initialized = False
        
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
