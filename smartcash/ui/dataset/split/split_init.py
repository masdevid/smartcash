"""
File: smartcash/ui/dataset/split/split_init.py
Deskripsi: Independent configuration cell for dataset split configuration with defaults integration
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from smartcash.common.logger import get_logger
from smartcash.ui.initializers.config_cell_initializer import create_config_cell
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
from smartcash.ui.dataset.split.handlers.config_extractor import extract_split_config
from smartcash.ui.dataset.split.handlers.config_updater import update_split_ui, reset_ui_to_defaults


class SplitConfigHandler(ConfigCellHandler):
    """Handler for split configuration with integrated defaults and validation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with default config and update with provided values"""
        super().__init__(module_name="split_config")
        # Initialize with default config
        self.config = get_default_split_config()
        # Update with any provided config values
        if config:
            self.update_config(config)
        # Save the initial config
        self.save()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        # Update config and notify listeners
        self.config.update(new_config)
        self.save()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        self.config = get_default_split_config()
        self.save()
    
    def update_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration from UI components
        
        Args:
            ui_components: Dictionary of UI components containing the configuration
            
        Returns:
            Dictionary containing the extracted configuration
        """
        try:
            # Extract configuration from UI components
            config = extract_split_config(ui_components)
            self.update_config(config)
            return config
            
        except Exception as e:
            logger = get_logger('smartcash.ui.dataset.split')
            logger.error(f"Failed to update config from UI: {str(e)}", exc_info=True)
            raise


def create_split_config_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create and return the split configuration UI components.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of UI components
    """
    try:
        from .components.ui_form import create_split_form
        from .components.ui_layout import create_split_layout
        from .handlers.slider_handlers import setup_slider_handlers
        
        # Create form components with the provided config
        form_components = create_split_form(config or {})
        
        # Create layout with form components
        layout_components = create_split_layout(form_components)
        
        # Combine all components
        ui_components = {**form_components, **layout_components}
        
        # Setup custom slider handlers
        setup_slider_handlers(ui_components)
        
        # Update UI with config values if provided
        if config:
            update_split_ui(ui_components, config)
            
        return ui_components
        
    except Exception as e:
        logger = get_logger('smartcash.ui.dataset.split')
        error_msg = f"Error creating split config UI: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        # Create a minimal error UI
        import ipywidgets as widgets
        from IPython.display import display
        
        error_output = widgets.Output()
        with error_output:
            print(error_msg)
            
        error_ui = {
            'error_output': error_output,
            'container': widgets.VBox([widgets.HTML('<h3>Error in Split Configuration UI</h3>'), error_output])
        }
        display(error_ui['container'])
        return error_ui


def create_split_config_cell(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create an independent split configuration cell
    
    Args:
        config: Optional initial configuration (merged with defaults)
        
    Returns:
        Dictionary containing UI components and handlers
    """
    logger = get_logger('smartcash.ui.dataset.split')
    
    try:
        # Create handler with default config and update with provided values
        handler = SplitConfigHandler(config)
        
        # Initialize the config cell
        ui_components = create_config_cell(
            module_name="split_config",
            config_filename="split_config",
            config_handler_class=lambda c=handler.config: SplitConfigHandler(c),
            config=handler.config  # Use handler.config directly since it's now inherited
        )
        
        # Create the split config UI with the current config
        split_ui = create_split_config_ui(handler.config)
        
        # Connect the UI to the config handler
        ui_components.update(split_ui)
        
        # Add reset handler if reset button exists
        if 'reset_button' in ui_components:
            def on_reset_clicked(b):
                handler.reset_to_defaults()
                reset_ui_to_defaults(ui_components)
                ui_components['_config_handler'].update_config(handler.config)
            
            ui_components['reset_button'].on_click(on_reset_clicked)
        
        return ui_components
        
    except Exception as e:
        import traceback
        error_msg = f"Failed to initialize Split Config UI: {str(e)}"
        error_traceback = traceback.format_exc()
        
        # Create and display error component
        from smartcash.ui.components.error.error_component import ErrorComponent
        error_component = ErrorComponent(title="Split Config Installer Error")
        error_ui = error_component.create(
            error_message=error_msg,
            traceback=error_traceback,
            error_type="error",
            show_traceback=True
        )
        display(error_ui['widget'])
        return error_ui