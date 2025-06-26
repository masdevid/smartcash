"""
File: smartcash/ui/dataset/split/split_init.py

This module implements the dataset split configuration UI by extending ConfigCellInitializer.
It follows the template method pattern where the parent class handles common initialization,
component registration, and error handling, while this class implements specific UI components
and business logic for dataset splitting. All errors are handled through the parent's
centralized error handling system for consistent user feedback.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.utils import safe_display
import logging
from IPython.display import display


# Initializers
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.config_cell.components.component_registry import component_registry

# Local components
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler

logger = logging.getLogger(__name__)

# Constants
MODULE_NAME = "split_config"

class SplitConfigInitializer(ConfigCellInitializer):
    """Initialize the dataset split configuration UI components."""
    
    def __init__(self):
        """Initialize the split config initializer."""
        super().__init__(
            module_name=MODULE_NAME,
            config_filename=MODULE_NAME,
            is_container=True  # We need a container for the UI
        )
        
    def create_handler(self, config: Optional[Dict[str, Any]] = None) -> SplitConfigHandler:
        """Create and return a SplitConfigHandler instance.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Initialized SplitConfigHandler instance
        """
        return SplitConfigHandler(config or {})
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create the UI components for the dataset split configuration.
        
        Returns:
            Dictionary containing the UI components with a 'container' widget.
        """
        # Create form components
        ui_components = create_split_form(config or {})
        
        # Create the main layout
        layout_components = create_split_layout(ui_components)
        ui_components.update(layout_components)
        
        # Ensure we have a container
        if 'container' not in ui_components:
            ui_components['container'] = widgets.VBox()
            
        # Get the main container
        container = ui_components['container']
        
        # Add all components to the container if it's a container widget
        if hasattr(container, 'children'):
            # Get all widgets that should be in the container
            widgets_to_add = [
                widget for key, widget in ui_components.items() 
                if key != 'container' and isinstance(widget, widgets.Widget)
            ]
            container.children = tuple(widgets_to_add)
        
        return ui_components
    
    def setup_handlers(self) -> None:
        """Set up event handlers for the UI components.
        
        This method is called after UI components are created and can be overridden
        to set up any additional event handlers, observers, or callbacks needed
        for the UI to function properly.
        """
        try:
            # Call parent implementation first
            super().setup_handlers()
            
            # Set up our custom event handlers
            from .handlers.event_handlers import setup_event_handlers
            setup_event_handlers(self, self.ui_components)
            
            self.logger.debug("Successfully set up event handlers for dataset split UI")
            
        except Exception as e:
            self.logger.error(f"Failed to set up event handlers: {str(e)}", exc_info=True)
            raise

def create_split_config_cell(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create and display a standalone split configuration container.
    
    This function initializes the split configuration UI and displays it in the notebook.
    The returned UI components dictionary includes all interactive elements and can be
    used to programmatically interact with the UI.
    
    Args:
        config: Optional configuration dictionary to override defaults. If not provided,
               default values will be used.
               
    Returns:
        An ipywidgets.Widget instance that can be displayed.
    """
    try:
        # Initialize the split config with the provided config
        initializer = SplitConfigInitializer()
        
        # Get the UI components
        ui_components = initializer.create_ui_components(config or {})
        
        # Get the container widget
        container = ui_components.get('container')
        if not container:
            container = widgets.VBox()
            ui_components['container'] = container
            
        # Set up handlers
        initializer.setup_handlers()
        
        # Display the container
        display(container)
        
        # Return all components for programmatic access
        return ui_components
        
    except Exception as e:
        error_msg = f"Failed to create split config cell: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Use the parent class's error handler for consistent error display
        from smartcash.ui.config_cell.handlers.error_handler import create_error_response
        # Centralized error handler
        return create_error_response(
            error_message=error_msg,
            error=e,
            title="Error in Dataset Split Configuration",
            include_traceback=True
        )
