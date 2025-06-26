"""
File: smartcash/ui/dataset/split/split_init.py

This module provides a configuration interface for dataset splitting that integrates
with the config cell initialization system. It handles the specific UI and business
logic for dataset split configuration while leveraging the common infrastructure
from config_cell_initializer.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
import logging

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.dataset.split.components.ui_form import create_split_form
from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
from smartcash.ui.dataset.split.handlers.config_updater import update_split_ui, reset_ui_to_defaults
from smartcash.ui.dataset.split.handlers.defaults import validate_split_ratios

logger = logging.getLogger(__name__)

class SplitConfigInitializer(ConfigCellInitializer):
    """Initialize the dataset split configuration UI components."""
    
    def __init__(self, parent_module: str = None):
        """Initialize the split config initializer.
        
        Args:
            parent_module: Optional parent module name for component registration
        """
        super().__init__(
            module_name="dataset_split",
            config_filename="dataset_split_config",
            parent_module=parent_module,
            is_container=True  # Mark as container to avoid default UI components
        )
        self.handler = None
        
    def create_handler(self, config: Dict[str, Any]) -> SplitConfigHandler:
        """Create and return a SplitConfigHandler instance."""
        return SplitConfigHandler(config)
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create the UI components for the dataset split configuration.
        
        Returns:
            Dictionary containing the UI components with 'container' as the root widget
        """
        try:
            # Create form components
            ui_components = create_split_form(config or {})
            
            # Create the main layout
            layout_components = create_split_layout(ui_components)
            ui_components.update(layout_components)
            
            # Set up event handlers
            self._setup_event_handlers(ui_components)
            
            return ui_components
            
        except Exception as e:
            logger.error(f"Error creating UI components: {e}")
            # Return a minimal error UI
            return {
                'container': widgets.HTML(
                    f"<div style='color: red; padding: 10px;'>"
                    f"Error initializing split configuration: {str(e)}</div>"
                )
            }
    
    def _setup_event_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Set up event handlers for the UI components.
        
        Args:
            ui_components: Dictionary of UI components
        """
        # Connect slider changes to update the total
        for slider in ['train_slider', 'valid_slider', 'test_slider']:
            if slider in ui_components:
                ui_components[slider].observe(
                    lambda change: self._on_slider_change(ui_components),
                    names='value'
                )
        
        # Connect save button
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(
                lambda b: self._on_save_clicked(ui_components)
            )
        
        # Connect reset button
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(
                lambda b: self._on_reset_clicked(ui_components)
            )
    
    def _on_slider_change(self, ui_components: Dict[str, Any]) -> None:
        """Handle slider value changes and update the total.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            total = sum([
                ui_components['train_slider'].value,
                ui_components['valid_slider'].value,
                ui_components['test_slider'].value
            ])
            
            # Update the total label
            if 'total_label' in ui_components:
                color = "#28a745" if abs(total - 1.0) < 0.001 else "#dc3545"
                ui_components['total_label'].value = (
                    f"<div style='padding: 10px; color: {color}; font-weight: bold;'>"
                    f"Total: {total:.2f} {'✓' if abs(total - 1.0) < 0.001 else '✗'}</div>"
                )
                
        except Exception as e:
            logger.error(f"Error updating slider total: {e}")
    
    def _on_save_clicked(self, ui_components: Dict[str, Any]) -> None:
        """Handle save button click.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Validate the split ratios
            if not validate_split_ratios(ui_components):
                self._update_status(ui_components, "Error: Split ratios must sum to 1.0", "danger")
                return
                
            # Update the config
            self.handler.update_config({
                'data': {
                    'split_ratios': {
                        'train': ui_components['train_slider'].value,
                        'valid': ui_components['valid_slider'].value,
                        'test': ui_components['test_slider'].value
                    },
                    'stratified_split': ui_components['stratified_checkbox'].value,
                    'random_seed': ui_components['random_seed'].value
                },
                'split_settings': {
                    'backup_before_split': ui_components['backup_checkbox'].value,
                    'dataset_path': ui_components['dataset_path'].value,
                    'preprocessed_path': ui_components['preprocessed_path'].value,
                    'backup_dir': ui_components['backup_dir'].value
                }
            })
            
            # Save the config
            self.handler.save_config()
            self._update_status(ui_components, "Configuration saved successfully!", "success")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            self._update_status(ui_components, f"Error saving configuration: {str(e)}", "danger")
    
    def _on_reset_clicked(self, ui_components: Dict[str, Any]) -> None:
        """Handle reset button click.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Reset to default values
            reset_ui_to_defaults(ui_components)
            
            # Update the UI with the default values
            update_split_ui(ui_components, self.handler.get_default_config())
            
            self._update_status(ui_components, "Reset to default values", "info")
            
        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")
            self._update_status(ui_components, f"Error resetting to defaults: {str(e)}", "danger")
    
    def _update_status(self, ui_components: Dict[str, Any], message: str, status_type: str) -> None:
        """Update the status message.
        
        Args:
            ui_components: Dictionary of UI components
            message: Status message to display
            status_type: Type of status (success, danger, info, warning)
        """
        # Log the status message
        log_level = {
            'success': logging.INFO,
            'danger': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO
        }.get(status_type, logging.INFO)
        
        logger.log(log_level, message)
        
        # If there's a status panel, update it
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = message
            ui_components['status_panel'].style = {
                'description_width': 'initial',
                'font_weight': 'bold',
                'color': {
                    'success': '#28a745',
                    'danger': '#dc3545',
                    'warning': '#ffc107',
                    'info': '#17a2b8'
                }.get(status_type, '#17a2b8')
            }


def create_split_config_cell(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create and display a standalone split configuration container.
    
    This function initializes the split configuration UI and displays it in the notebook.
    The returned UI components dictionary includes all interactive elements and can be
    used to programmatically interact with the UI.
    
    Args:
        config: Optional configuration dictionary to override defaults. If not provided,
               default values will be used.
               
    Returns:
        Dictionary containing the UI components with 'container' as the root widget.
        The dictionary includes all interactive components for programmatic access.
    """
    try:
        # Initialize the split config
        initializer = SplitConfigInitializer()
        
        # Create handler with the provided or default config
        handler = initializer.create_handler(config or {})
        initializer.handler = handler
        
        # Create and initialize UI components
        ui_components = initializer.create_ui_components(handler.config)
        
        # Store references for later use
        initializer.ui_components = ui_components
        
        # Display the container if it exists
        container = ui_components.get('container')
        if container is not None:
            display(container)
        else:
            logger.warning("No container found in UI components")
        
        return ui_components
        
    except Exception as e:
        error_msg = f"Failed to create split config cell: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Create and display an error message
        error_widget = widgets.HTML(
            f"<div style='color: red; padding: 10px; border: 1px solid #f5c6cb; border-radius: 4px;'>"
            f"<strong>Error:</strong> {error_msg}</div>"
        )
        display(error_widget)
        
        return {'container': error_widget, 'error': str(e)}
