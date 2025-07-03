"""
File: smartcash/ui/dataset/split/split_initializer.py
Deskripsi: Implementation of dataset split configuration UI following CommonInitializer pattern
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
import logging

# Import CommonInitializer
from smartcash.ui.initializers.common_initializer import CommonInitializer

# Import local components - HANYA form components
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout

# Import handlers
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors, create_error_response

# Constants
MODULE_NAME = "split_config"

class SplitInitializer(CommonInitializer):
    """Dataset split configuration UI implementation following CommonInitializer pattern.
    
    Features:
    - Follows CommonInitializer pattern for consistent initialization flow
    - Centralized error handling and logging
    - Config load/save functionality without progress tracking
    - No summary panel, only logging
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize split config UI.
        
        Args:
            config: Configuration dictionary untuk split settings
            **kwargs: Additional arguments untuk parent class
        """
        # Initialize CommonInitializer with module name and config handler class
        super().__init__(
            module_name=MODULE_NAME,
            config_handler_class=SplitConfigHandler
        )
        
        # Store initial config
        self.initial_config = config or {}
        
        # Store additional kwargs
        self.kwargs = kwargs
        
        # UI components dictionary
        self.ui_components = {}
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create and return UI components as a dictionary.
        
        Args:
            config: Loaded configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        # Create UI components
        ui_components = {}
        
        # Create main container
        container = widgets.VBox()
        
        # Create header
        header = widgets.HTML(
            f"<h2>Dataset Split Configuration</h2>"
            f"<p>Konfigurasi pembagian dataset untuk training, validation, dan testing</p>"
        )
        
        # Create log output
        log_output = widgets.Output()
        log_accordion = widgets.Accordion(children=[log_output])
        log_accordion.set_title(0, "Log")
        
        # Create form components
        form_components = create_split_form(config)
        
        # Create save and reset buttons
        save_button = widgets.Button(
            description="Save",
            button_style="primary",
            icon="save"
        )
        
        reset_button = widgets.Button(
            description="Reset",
            button_style="warning",
            icon="refresh"
        )
        
        # Create button container
        button_container = widgets.HBox([save_button, reset_button])
        
        # Create layout
        layout = create_split_layout(
            form_components=form_components,
            button_container=button_container
        )
        
        # Assemble container
        container.children = [
            header,
            layout,
            log_accordion
        ]
        
        # Store components in dictionary
        ui_components = {
            # Main container
            'container': container,
            'ui': container,  # Alias for compatibility
            
            # Header
            'header': header,
            
            # Log components
            'log_output': log_output,
            'log_accordion': log_accordion,
            
            # Buttons
            'save_button': save_button,
            'reset_button': reset_button,
            'button_container': button_container,
            
            # Form components
            **form_components
        }
        
        # Set handler's UI components
        if hasattr(self.config_handler, 'set_ui_components'):
            self.config_handler.set_ui_components(ui_components)
        
        # Store UI components
        self.ui_components = ui_components
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for this module.
        
        Returns:
            Default configuration dictionary
        """
        if hasattr(self.config_handler, 'get_default_config'):
            return self.config_handler.get_default_config()
        return {}
    
    def _setup_handlers(self) -> None:
        """Set up event handlers for UI components.
        
        This method connects UI components to their respective handlers.
        """
        if not self.ui_components:
            return
            
        # Get UI components
        save_button = self.ui_components.get('save_button')
        reset_button = self.ui_components.get('reset_button')
        
        # Set up save button handler
        if save_button and hasattr(self.config_handler, 'save_config'):
            save_button.on_click(
                lambda b: self.config_handler.save_config(self.ui_components)
            )
            
        # Set up reset button handler
        if reset_button and hasattr(self.config_handler, 'reset_ui'):
            reset_button.on_click(
                lambda b: self.config_handler.reset_ui(self.ui_components)
            )
            
        # Set up slider handlers if available
        try:
            from smartcash.ui.dataset.split.handlers.slider_handlers import setup_slider_handlers
            setup_slider_handlers(self.ui_components)
        except Exception as e:
            self.logger.warning(f"Failed to set up slider handlers: {str(e)}")
            
        # Set up event handlers if available
        try:
            from smartcash.ui.dataset.split.handlers.event_handlers import setup_event_handlers
            setup_event_handlers(self.ui_components)
        except Exception as e:
            self.logger.warning(f"Failed to set up event handlers: {str(e)}")

@handle_ui_errors(error_component_title="Split Config Error", log_error=True)
def create_split_config_cell(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Create dan display dataset split configuration UI.
    
    PUBLIC API FUNCTION - Main entry point untuk users.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments untuk initializer
        
    Returns:
        None - UI is displayed directly
    """
    # Create initializer
    initializer = SplitInitializer(config=config, **kwargs)
    
    # Initialize UI
    ui = initializer.initialize()
    
    # Display UI
    display(ui)

@handle_ui_errors(error_component_title="Split Config Error", log_error=True)
def get_split_config_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create split config UI dan return components untuk programmatic access.
    
    PUBLIC API FUNCTION - Untuk programmatic access ke components.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments untuk initializer
               
    Returns:
        Dictionary berisi semua UI components
    """
    # Create initializer
    initializer = SplitInitializer(config=config, **kwargs)
    
    # Initialize UI without displaying
    initializer.initialize()
    
    # Return components dictionary
    return {
        # Core references
        'initializer': initializer,
        'handler': initializer.config_handler,
        'logger': initializer.logger,
        
        # Main container
        'container': initializer.ui_components.get('container'),
        
        # All UI components
        **initializer.ui_components
    }
