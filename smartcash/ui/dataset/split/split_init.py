"""
File: smartcash/ui/dataset/split/split_init.py
Deskripsi: Independent configuration cell for dataset split configuration with defaults integration
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import yaml
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.initializers.config_cell_initializer import create_config_cell
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
from smartcash.ui.dataset.split.handlers.config_extractor import extract_split_config
from smartcash.ui.dataset.split.handlers.config_updater import update_split_ui, reset_ui_to_defaults
from smartcash.ui.utils.logging_utils import suppress_all_outputs, restore_stdout


class SplitConfigHandler(ConfigCellHandler):
    """Handler for split configuration with integrated defaults and validation"""
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_bridge: Optional[Callable] = None):
        """
        Initialize with default config and update with provided values
        
        Args:
            config: Optional configuration dictionary
            logger_bridge: Optional logger bridge function for UI logging
        """
        # Initialize logger bridge first to capture any initialization logs
        self.logger_bridge = logger_bridge or get_logger('smartcash.ui.dataset.split')
        
        # Suppress output during initialization
        with suppress_all_outputs():
            super().__init__(module_name="split_config")
            # Initialize with default config
            self.config = get_default_split_config()
            # Update with any provided config values
            if config:
                self.update_config(config)
            # Save the initial config
            self.save()
            
            # Log successful initialization
            if hasattr(self.logger_bridge, '__call__'):
                self.logger_bridge("SplitConfigHandler initialized successfully", level='info')
            else:
                self.logger_bridge.info("SplitConfigHandler initialized successfully")
    
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
            # Try to use logger_bridge if available, otherwise fall back to standard logger
            logger = getattr(self, 'logger_bridge', None) or get_logger('smartcash.ui.dataset.split')
            error_msg = f"Failed to update config from UI: {str(e)}"
            if hasattr(logger, '__call__'):
                logger(error_msg, level='error')
            else:
                logger.error(error_msg, exc_info=True)
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
        restore_stdout()  # Ensure output is restored even on error
        error_fallback = _create_error_fallback(str(e))
        if 'container' in error_fallback:
            return error_fallback['container']
        return error_fallback


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
        restore_stdout()  # Ensure output is restored even on error
        error_fallback = _create_error_fallback(str(e))
        if 'container' in error_fallback:
            return error_fallback['container']
        return error_fallback

def _create_error_fallback(error_message: str, traceback: Optional[str] = None) -> widgets.VBox:
    """Create a fallback UI component to display error messages."""
    from smartcash.ui.components import create_error_component
    return create_error_component("Initialization Error", error_message, traceback)
        