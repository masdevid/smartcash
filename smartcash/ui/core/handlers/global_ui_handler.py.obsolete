"""
Global UI Handler for Save/Reset and Header Status Management

This module provides centralized management for Save/Reset functionality
and header status updates that applies to all UI modules in SmartCash.
"""

from typing import Dict, Any, Optional, Callable, Union
from smartcash.common.logger import get_logger
from smartcash.ui.components.header_container import HeaderContainer
from smartcash.ui.core.decorators import handle_ui_errors


class GlobalUIHandler:
    """Global handler for Save/Reset functionality and header status updates."""
    
    def __init__(self, module_name: str = "global_ui"):
        """Initialize the global UI handler.
        
        Args:
            module_name: Name of the module using this handler
        """
        self.module_name = module_name
        self.logger = get_logger(f"ui.{module_name}")
        self.header_container = None
        self.save_callback = None
        self.reset_callback = None
        self.form_widgets = {}
        self.default_values = {}
        
    def set_header_container(self, header_container: HeaderContainer):
        """Set the header container for status updates.
        
        Args:
            header_container: The HeaderContainer instance to update
        """
        self.header_container = header_container
        
    def set_form_widgets(self, form_widgets: Dict[str, Any]):
        """Set the form widgets to be managed.
        
        Args:
            form_widgets: Dictionary of form widgets
        """
        self.form_widgets = form_widgets
        
    def set_default_values(self, default_values: Dict[str, Any]):
        """Set the default values for form reset.
        
        Args:
            default_values: Dictionary of default values
        """
        self.default_values = default_values
        
    def set_save_callback(self, callback: Callable[[Dict[str, Any]], bool]):
        """Set the callback function for save operations.
        
        Args:
            callback: Function that takes form values and returns success status
        """
        self.save_callback = callback
        
    def set_reset_callback(self, callback: Optional[Callable[[], bool]] = None):
        """Set the callback function for reset operations.
        
        Args:
            callback: Optional function for custom reset logic
        """
        self.reset_callback = callback
        
    def update_header_status(self, message: str, status_type: str = "info", show: bool = True):
        """Update the header status panel.
        
        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error)
            show: Whether to show the status panel
        """
        if self.header_container:
            try:
                self.header_container.update_status(message, status_type, show)
                self.logger.debug(f"Header status updated: {message} ({status_type})")
            except Exception as e:
                self.logger.error(f"Failed to update header status: {str(e)}")
                
    def get_form_values(self) -> Dict[str, Any]:
        """Extract current values from form widgets.
        
        Returns:
            Dictionary of current form values
        """
        values = {}
        for widget_name, widget in self.form_widgets.items():
            if widget is None:
                continue
                
            try:
                if hasattr(widget, 'value'):
                    values[widget_name] = widget.value
                elif hasattr(widget, 'selected'):
                    values[widget_name] = widget.selected
                elif hasattr(widget, 'description'):
                    values[widget_name] = widget.description
            except Exception as e:
                self.logger.warning(f"Failed to get value for {widget_name}: {str(e)}")
                values[widget_name] = None
                
        return values
        
    def set_form_values(self, values: Dict[str, Any]):
        """Set values in form widgets.
        
        Args:
            values: Dictionary of values to set
        """
        for widget_name, value in values.items():
            widget = self.form_widgets.get(widget_name)
            if widget is None or value is None:
                continue
                
            try:
                if hasattr(widget, 'value'):
                    widget.value = value
                elif hasattr(widget, 'selected'):
                    widget.selected = value
                elif hasattr(widget, 'description'):
                    widget.description = str(value)
            except Exception as e:
                self.logger.warning(f"Failed to set value for {widget_name}: {str(e)}")
                
    @handle_ui_errors(error_component_title="Save Operation Error")
    def handle_save(self) -> bool:
        """Handle save operation with status updates.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Update status to indicate saving
            self.update_header_status("💾 Saving configuration...", "info")
            
            # Get current form values
            form_values = self.get_form_values()
            
            # Call save callback if provided
            if self.save_callback:
                success = self.save_callback(form_values)
            else:
                # Default save behavior - just log the values
                self.logger.info(f"Configuration saved: {form_values}")
                success = True
                
            # Update status based on result
            if success:
                self.update_header_status("✅ Configuration saved successfully", "success")
                self.logger.info(f"🎉 {self.module_name} configuration saved successfully")
                return True
            else:
                self.update_header_status("❌ Failed to save configuration", "error")
                self.logger.error(f"❌ Failed to save {self.module_name} configuration")
                return False
                
        except Exception as e:
            error_msg = f"Save operation failed: {str(e)}"
            self.update_header_status(f"❌ {error_msg}", "error")
            self.logger.error(f"❌ {self.module_name} save error: {error_msg}")
            return False
            
    @handle_ui_errors(error_component_title="Reset Operation Error")
    def handle_reset(self) -> bool:
        """Handle reset operation with status updates.
        
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            # Update status to indicate resetting
            self.update_header_status("🔄 Resetting to defaults...", "info")
            
            # Call custom reset callback if provided
            if self.reset_callback:
                success = self.reset_callback()
                if not success:
                    self.update_header_status("❌ Failed to reset configuration", "error")
                    return False
            
            # Default reset behavior - set form values to defaults
            if self.default_values:
                self.set_form_values(self.default_values)
                
            # Update status to indicate success
            self.update_header_status("🔄 Configuration reset to defaults", "success")
            self.logger.info(f"🔄 {self.module_name} configuration reset to defaults")
            return True
            
        except Exception as e:
            error_msg = f"Reset operation failed: {str(e)}"
            self.update_header_status(f"❌ {error_msg}", "error")
            self.logger.error(f"❌ {self.module_name} reset error: {error_msg}")
            return False
            
    def bind_save_reset_buttons(self, save_button, reset_button):
        """Bind the save and reset button click handlers.
        
        Args:
            save_button: The save button widget
            reset_button: The reset button widget
        """
        if save_button and hasattr(save_button, 'on_click'):
            save_button.on_click(lambda _: self.handle_save())
            
        if reset_button and hasattr(reset_button, 'on_click'):
            reset_button.on_click(lambda _: self.handle_reset())
            
        self.logger.debug("Save/Reset button handlers bound successfully")


def create_global_ui_handler(
    module_name: str,
    header_container: Optional[HeaderContainer] = None,
    form_widgets: Optional[Dict[str, Any]] = None,
    default_values: Optional[Dict[str, Any]] = None,
    save_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    reset_callback: Optional[Callable[[], bool]] = None
) -> GlobalUIHandler:
    """Create and configure a global UI handler.
    
    Args:
        module_name: Name of the module
        header_container: Optional header container for status updates
        form_widgets: Optional form widgets to manage
        default_values: Optional default values for reset
        save_callback: Optional save operation callback
        reset_callback: Optional reset operation callback
        
    Returns:
        Configured GlobalUIHandler instance
    """
    handler = GlobalUIHandler(module_name)
    
    if header_container:
        handler.set_header_container(header_container)
    if form_widgets:
        handler.set_form_widgets(form_widgets)
    if default_values:
        handler.set_default_values(default_values)
    if save_callback:
        handler.set_save_callback(save_callback)
    if reset_callback:
        handler.set_reset_callback(reset_callback)
        
    return handler