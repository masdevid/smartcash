"""
File: smartcash/ui/dataset/split/handlers/event_handlers.py

Event handlers for the dataset split configuration UI.
"""

import logging
from typing import Dict, Any, Callable
import ipywidgets as widgets

from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
from smartcash.ui.dataset.split.handlers.config_updater import update_split_ui, reset_ui_to_defaults

logger = logging.getLogger(__name__)


def setup_event_handlers(initializer, ui_components: Dict[str, Any]) -> None:
    """Set up event handlers for the UI components.
    
    Args:
        initializer: The SplitConfigInitializer instance
        ui_components: Dictionary of UI components
    """
    # Connect slider changes to update the total
    for slider in ['train_slider', 'valid_slider', 'test_slider']:
        if slider in ui_components:
            ui_components[slider].observe(
                lambda change, ui=ui_components: on_slider_change(initializer, ui),
                names='value'
            )
    
    # Connect save button
    save_reset = ui_components.get('save_reset_buttons', {})
    if 'save_button' in save_reset:
        save_reset['save_button'].on_click(
            lambda b, ui=ui_components: on_save_clicked(initializer, ui)
        )
    
    # Connect reset button
    if 'reset_button' in save_reset:
        save_reset['reset_button'].on_click(
            lambda b, ui=ui_components: on_reset_clicked(initializer, ui)
        )


def on_slider_change(initializer, ui_components: Dict[str, Any]) -> None:
    """Handle slider value changes and update the total.
    
    Args:
        initializer: The SplitConfigInitializer instance
        ui_components: Dictionary of UI components
    """
    try:
        train = ui_components['train_slider'].value
        valid = ui_components['valid_slider'].value
        test = ui_components['test_slider'].value
        
        # Validate the ratios using the handler
        is_valid = initializer.handler.validate_split_ratios(train, valid, test)
        
        # Update the total label
        if 'total_label' in ui_components:
            color = "#28a745" if is_valid else "#dc3545"
            total = train + valid + test
            ui_components['total_label'].value = (
                f"<div style='padding: 10px; color: {color}; font-weight: bold;'>"
                f"Total: {total:.2f} {'✓' if is_valid else '✗'}</div>"
            )
            
    except Exception as e:
        logger.error(f"Error updating slider total: {e}")


def on_save_clicked(initializer, ui_components: Dict[str, Any]) -> None:
    """Handle save button click.
    
    Args:
        initializer: The SplitConfigInitializer instance
        ui_components: Dictionary of UI components
    """
    try:
        # Validate the split ratios using the handler
        train = ui_components['train_slider'].value
        valid = ui_components['valid_slider'].value
        test = ui_components['test_slider'].value
        
        if not initializer.handler.validate_split_ratios(train, valid, test):
            initializer._update_status(ui_components, "Error: Split ratios must sum to 1.0", "danger")
            return
            
        # Update the config
        initializer.handler.update_config({
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
        initializer.handler.save_config()
        initializer._update_status(ui_components, "Configuration saved successfully!", "success")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        initializer._update_status(ui_components, f"Error saving configuration: {str(e)}", "danger")


def on_reset_clicked(initializer, ui_components: Dict[str, Any]) -> None:
    """Handle reset button click.
    
    Args:
        initializer: The SplitConfigInitializer instance
        ui_components: Dictionary of UI components
    """
    try:
        # Reset to default values
        reset_ui_to_defaults(ui_components)
        
        # Update the UI with the default values
        update_split_ui(ui_components, initializer.handler.get_default_config())
        
        initializer._update_status(ui_components, "Reset to default values", "info")
        
    except Exception as e:
        logger.error(f"Error resetting to defaults: {e}")
        initializer._update_status(ui_components, f"Error resetting to defaults: {str(e)}", "danger")
