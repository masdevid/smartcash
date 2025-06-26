"""
File: smartcash/ui/dataset/split/split_init.py

This module provides a configuration interface for dataset splitting that integrates
with the config cell initialization system. It handles the specific UI and business
logic for dataset split configuration while leveraging the common infrastructure
from config_cell_initializer.
"""

from typing import Dict, Any, Optional, Type, List
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
from smartcash.ui.dataset.split.handlers.config_extractor import extract_split_config
from smartcash.ui.dataset.split.handlers.config_updater import update_split_ui, reset_ui_to_defaults
from smartcash.ui.utils.logging_utils import suppress_all_outputs, restore_stdout

class SplitConfigHandler(ConfigCellHandler):
    """Handler for dataset split configuration.
    
    This extends the base ConfigCellHandler with dataset split specific
    configuration handling and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize with default split configuration."""
        super().__init__(module_name="split_config", **kwargs)
        self.config = get_default_split_config()
        if config:
            self.update_config(config)
    
    def update_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration from UI components."""
        config = extract_split_config(ui_components)
        self.update_config(config)
        return config
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = get_default_split_config()
        self.save()


def create_split_config_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create UI components for split configuration."""
    from smartcash.ui.dataset.split.components.ui_form import create_split_form
    from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
    from smartcash.ui.dataset.split.handlers.slider_handlers import setup_slider_handlers
    
    # Create form and layout components
    form_components = create_split_form(config or {})
    layout_components = create_split_layout(form_components)
    
    # Combine all components
    ui_components = {**form_components, **layout_components}
    
    # Ensure we have a container
    if 'container' not in ui_components:
        ui_components['container'] = widgets.VBox()
    
    # Setup custom handlers
    setup_slider_handlers(ui_components)
    
    # Update UI with config values
    if config:
        update_split_ui(ui_components, config)
    
    return ui_components


class SplitConfigInitializer(ConfigCellInitializer[SplitConfigHandler]):
    """Initializer for dataset split configuration UI."""
    
    def __init__(self):
        super().__init__(
            module_name="split_config",
            config_filename="split_config"
        )
        self.logger = get_logger('smartcash.ui.dataset.split.initializer')
    
    def create_handler(self) -> SplitConfigHandler:
        """Create and return a new SplitConfigHandler instance."""
        return SplitConfigHandler()
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return UI components for split configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict containing UI components with 'container' as the root widget
        """
        # Create UI components
        ui_components = create_split_config_ui(config or {})
        
        # Get the main container from components or create a new one
        main_container = ui_components.get('main_container')
        if not isinstance(main_container, widgets.Widget):
            main_container = widgets.VBox()
            
        # Style the main container
        main_container.layout = widgets.Layout(
            width='100%',
            padding='10px',
            border='1px solid #e0e0e0',
            margin='5px 0',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            overflow='visible'
        )
        
        # Ensure all children are valid widgets
        if hasattr(main_container, 'children'):
            valid_children = [
                child for child in main_container.children 
                if isinstance(child, widgets.Widget)
            ]
            main_container.children = valid_children
        
        # Update the container in ui_components
        ui_components['container'] = main_container
            
        return ui_components
    
    def setup_handlers(self) -> None:
        """Set up event handlers for UI components."""
        if 'reset_button' in self.ui_components:
            def on_reset_clicked(b):
                with suppress_all_outputs():
                    self.handler.reset_to_defaults()
                    reset_ui_to_defaults(self.ui_components)
                    update_split_ui(self.ui_components, self.handler.config)
                    
                    if 'status' in self.ui_components:
                        self.ui_components['status'].value = "Reset to default values"
            
            self.ui_components['reset_button'].on_click(on_reset_clicked)


def create_split_config_cell(config: Optional[Dict[str, Any]] = None) -> None:
    """Create and display a split configuration cell.
    
    Args:
        config: Optional configuration to override defaults
    """
    initializer = SplitConfigInitializer()
    initializer.initialize(config or {})
