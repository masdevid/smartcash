"""
File: smartcash/ui/dataset/split/split_init.py

This module implements the dataset split configuration UI by extending ConfigCellInitializer.
It follows the template method pattern where the parent class handles common initialization,
component registration, and error handling, while this class implements specific UI components
and business logic for dataset splitting. All errors are handled through the parent's
centralized error handling system for consistent user feedback.
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display
import logging

# Initializers
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer

# Local components
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler

logger = logging.getLogger(__name__)

# Constants
MODULE_NAME = "split_config"

class SplitConfigInitializer(ConfigCellInitializer):
    """Initialize the dataset split configuration UI components."""
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        component_id: str = MODULE_NAME,
        title: str = "Dataset Split Configuration",
        children: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Initialize the split config initializer.
        
        Args:
            config: Optional configuration dictionary
            parent_id: Optional parent component ID
            component_id: Unique identifier for this component
            title: Display title for the component
            children: Optional list of child component configurations
            **kwargs: Additional keyword arguments passed to parent class
        """
        # Initialize the parent class first
        super().__init__(
            config=config or {},
            parent_id=parent_id,
            component_id=component_id,
            title=title,
            children=children or [],
            **kwargs
        )
        
        # Initialize the handler using the parent's _handler attribute
        self._handler = None
        
    def create_handler(self) -> SplitConfigHandler:
        """Create and return a SplitConfigHandler instance."""
        return SplitConfigHandler(self.config)
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create the UI components for the dataset split configuration.
        
        Returns:
            Dictionary containing the UI components. The parent class will handle
            adding these to the container.
        """
        try:
            self._logger.debug("Creating UI components for dataset split configuration")
            
            # Create form components
            ui_components = create_split_form(config or {})
            
            # Create the main layout
            layout_components = create_split_layout(ui_components)
            ui_components.update(layout_components)
            
            self._logger.debug("Successfully created UI components")
            return ui_components
            
        except Exception as e:
            self._logger.error(f"Failed to create UI components: {str(e)}", exc_info=True)
            # Return empty dict to let parent handle the error
            return {}
    
    def setup_handlers(self) -> None:
        """Set up event handlers for the UI components."""
        try:
            # Call parent implementation first
            super().setup_handlers()
            
            # Set up our custom event handlers
            from .handlers.event_handlers import setup_event_handlers
            setup_event_handlers(self, self.ui_components)
            
            self._logger.debug("Successfully set up event handlers for dataset split UI")
            
        except Exception as e:
            self._logger.error(f"Failed to set up event handlers: {str(e)}", exc_info=True)
            raise

def create_split_config_cell(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create and display a standalone split configuration container.
    
    This function initializes the split configuration UI and displays it in the notebook.
    The returned UI components dictionary includes all interactive elements and can be
    used to programmatically interact with the UI.
    
    Args:
        config: Optional configuration dictionary to override defaults.
        **kwargs: Additional arguments to pass to the initializer.
               
    Returns:
        Dictionary containing the UI components for programmatic access.
    """
    try:
        # Initialize the split config with the provided config and kwargs
        initializer = SplitConfigInitializer(config=config, **kwargs)
        
        # Initialize the UI (this will also register components and set up handlers)
        container = initializer.initialize()
        
        # Display the container
        display(container)
        
        # Return the UI components for programmatic access
        return {
            **initializer.ui_components,
            'container': initializer.parent_component.container,
            'content_area': initializer.parent_component.content_area
        }
        
    except Exception as e:
        error_msg = f"Failed to create split config cell: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Use the parent class's error handler for consistent error display
        from smartcash.ui.config_cell.handlers.error_handler import create_error_response
        error_widget = create_error_response(
            error_message=error_msg,
            error=e,
            title="Error in Dataset Split Configuration",
            include_traceback=True
        )
        display(error_widget)
        return {'error': error_widget}
