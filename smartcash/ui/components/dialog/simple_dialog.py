"""
Simple dialog component for displaying basic confirmation and info dialogs.

This component provides a clean, simple interface for showing dialogs without
complex animations, JavaScript, or CSS. It uses basic ipywidgets functionality
with auto-expanding/hiding confirmation areas.
"""

from typing import Dict, Callable, Optional
import ipywidgets as widgets

from smartcash.ui.components.base_component import BaseUIComponent


class SimpleDialog(BaseUIComponent):
    """Simple dialog component with basic hide/show functionality."""
    
    def __init__(self, component_name: str = "simple_dialog", **kwargs):
        """Initialize the simple dialog.
        
        Args:
            component_name: Unique name for this component
            **kwargs: Additional arguments to pass to BaseUIComponent
        """
        # Extract dialog-specific kwargs
        dialog_kwargs = {
            'title': kwargs.pop('title', ''),
            'message': kwargs.pop('message', ''),
            'on_confirm': kwargs.pop('on_confirm', None),
            'on_cancel': kwargs.pop('on_cancel', None),
            'confirm_text': kwargs.pop('confirm_text', 'Confirm'),
            'cancel_text': kwargs.pop('cancel_text', 'Cancel'),
            'danger_mode': kwargs.pop('danger_mode', False)
        }
        
        # Pass remaining kwargs to BaseUIComponent
        super().__init__(component_name, **kwargs)
        
        # Store dialog-specific kwargs
        self._dialog_kwargs = dialog_kwargs
        
        self._is_visible = False
        self._callbacks: Dict[str, Callable] = {}
        self._current_buttons: Dict[str, widgets.Button] = {}
    
    def _create_ui_components(self) -> None:
        """Create and initialize UI components."""
        # Main container that auto-expands/hides
        self._ui_components['container'] = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                margin='0',
                padding='0',
                display='none'  # Initially hidden
            )
        )
        
        # Content area for dialog messages
        self._ui_components['content'] = widgets.HTML(
            value="",
            layout=widgets.Layout(
                width='100%',
                margin='0 0 10px 0',
                padding='10px',
                border='1px solid #ddd',
                border_radius='4px',
                background='#f9f9f9'
            )
        )
        
        # Button area
        self._ui_components['button_area'] = widgets.HBox(
            layout=widgets.Layout(
                width='100%',
                margin='0',
                padding='0',
                justify_content='center'
            )
        )
        
        # Assemble the dialog
        self._ui_components['container'].children = [
            self._ui_components['content'],
            self._ui_components['button_area']
        ]
    
    @property
    def container(self):
        """Get the dialog container widget for display."""
        return self._ui_components.get('container')
    
    def show_confirmation(self, 
                         title: str,
                         message: str,
                         on_confirm: Optional[Callable] = None,
                         on_cancel: Optional[Callable] = None,
                         confirm_text: str = "Confirm",
                         cancel_text: str = "Cancel",
                         danger_mode: bool = False) -> None:
        """Show a confirmation dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_confirm: Callback for confirm action
            on_cancel: Callback for cancel action
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            danger_mode: Whether to use danger styling
        """
        if not self._initialized:
            self.initialize()
        
        # Store callbacks
        self._callbacks['confirm'] = on_confirm
        self._callbacks['cancel'] = on_cancel
        
        # Create content HTML
        border_color = "#dc3545" if danger_mode else "#007bff"
        
        content_html = f"""
        <div style="border-left: 4px solid {border_color}; padding: 0 10px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{title}</h4>
            <p style="margin: 0; color: #666;">{message}</p>
        </div>
        """
        
        # Update content
        self._ui_components['content'].value = content_html
        
        # Create buttons
        self._create_confirmation_buttons(confirm_text, cancel_text, danger_mode)
        
        # Show the dialog
        self._show_dialog()
    
    def show_info(self,
                  title: str,
                  message: str,
                  on_ok: Optional[Callable] = None,
                  ok_text: str = "OK",
                  info_type: str = "info") -> None:
        """Show an info dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback for OK action
            ok_text: Text for OK button
            info_type: Type of info dialog (info, success, warning, error)
        """
        if not self._initialized:
            self.initialize()
        
        # Store callback
        self._callbacks['ok'] = on_ok
        
        # Define colors based on info type
        colors = {
            'info': '#007bff',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
        
        color = colors.get(info_type, '#007bff')
        
        # Create content HTML
        content_html = f"""
        <div style="border-left: 4px solid {color}; padding: 0 10px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{title}</h4>
            <p style="margin: 0; color: #666;">{message}</p>
        </div>
        """
        
        # Update content
        self._ui_components['content'].value = content_html
        
        # Create OK button
        self._create_info_button(ok_text, info_type)
        
        # Show the dialog
        self._show_dialog()
    
    def _create_confirmation_buttons(self, confirm_text: str, cancel_text: str, danger_mode: bool) -> None:
        """Create confirmation dialog buttons."""
        # Clear existing buttons
        self._current_buttons.clear()
        
        # Create confirm button
        confirm_style = 'danger' if danger_mode else 'primary'
        confirm_button = widgets.Button(
            description=confirm_text,
            button_style=confirm_style,
            layout=widgets.Layout(margin='0 5px')
        )
        confirm_button.on_click(self._handle_confirm)
        self._current_buttons['confirm'] = confirm_button
        
        # Create cancel button
        cancel_button = widgets.Button(
            description=cancel_text,
            button_style='',
            layout=widgets.Layout(margin='0 5px')
        )
        cancel_button.on_click(self._handle_cancel)
        self._current_buttons['cancel'] = cancel_button
        
        # Update button area
        self._ui_components['button_area'].children = [
            confirm_button,
            cancel_button
        ]
    
    def _create_info_button(self, ok_text: str, info_type: str) -> None:
        """Create info dialog button."""
        # Clear existing buttons
        self._current_buttons.clear()
        
        # Create OK button
        button_style = {
            'info': 'primary',
            'success': 'success',
            'warning': 'warning',
            'error': 'danger'
        }.get(info_type, 'primary')
        
        ok_button = widgets.Button(
            description=ok_text,
            button_style=button_style,
            layout=widgets.Layout(margin='0 5px')
        )
        ok_button.on_click(self._handle_ok)
        self._current_buttons['ok'] = ok_button
        
        # Update button area
        self._ui_components['button_area'].children = [ok_button]
    
    def _handle_confirm(self, _button) -> None:
        """Handle confirm button click."""
        callback = self._callbacks.get('confirm')
        if callback:
            try:
                callback()
            except Exception as e:
                print(f"⚠️ Error in confirm callback: {e}")
        
        # Hide dialog after callback
        self.hide()
    
    def _handle_cancel(self, _button) -> None:
        """Handle cancel button click."""
        callback = self._callbacks.get('cancel')
        if callback:
            try:
                callback()
            except Exception as e:
                print(f"⚠️ Error in cancel callback: {e}")
        
        # Hide dialog after callback
        self.hide()
    
    def _handle_ok(self, _button) -> None:
        """Handle OK button click."""
        callback = self._callbacks.get('ok')
        if callback:
            try:
                callback()
            except Exception as e:
                print(f"⚠️ Error in OK callback: {e}")
        
        # Hide dialog after callback
        self.hide()
    
    def _show_dialog(self) -> None:
        """Show the dialog by making container visible."""
        container = self._ui_components['container']
        container.layout.display = 'block'
        self._is_visible = True
    
    def hide(self) -> None:
        """Hide the dialog."""
        if not self._initialized:
            return
        
        container = self._ui_components.get('container')
        if container:
            container.layout.display = 'none'
        
        # Clean up state
        self._is_visible = False
        self._callbacks.clear()
        self._current_buttons.clear()
    
    def is_visible(self) -> bool:
        """Check if the dialog is currently visible."""
        if not self._initialized:
            return False
        
        container = self._ui_components.get('container')
        if container:
            return container.layout.display != 'none'
        
        return self._is_visible
    
    def clear(self) -> None:
        """Clear the dialog content."""
        if not self._initialized:
            return
        
        self._ui_components['content'].value = ""
        self._ui_components['button_area'].children = []
        self.hide()
    
    # Additional callback management methods for test compatibility
    def _add_callback(self, key: str, callback: Callable) -> None:
        """Add a callback for a specific key.
        
        Args:
            key: Callback key identifier
            callback: Callback function to store
        """
        self._callbacks[key] = callback
    
    def _get_callback(self, key: str) -> Optional[Callable]:
        """Get a callback by key.
        
        Args:
            key: Callback key identifier
            
        Returns:
            The callback function if found, None otherwise
        """
        return self._callbacks.get(key)
    
    def _clear_callbacks(self) -> None:
        """Clear all stored callbacks."""
        self._callbacks.clear()


# Factory functions for easy usage
def create_simple_dialog(component_name: str = "dialog") -> SimpleDialog:
    """Create a simple dialog instance.
    
    Args:
        component_name: Name for the dialog component
        
    Returns:
        SimpleDialog: Configured dialog instance
    """
    dialog = SimpleDialog(component_name)
    dialog.initialize()
    return dialog


def show_confirmation_dialog(dialog: SimpleDialog,
                           title: str,
                           message: str,
                           on_confirm: Optional[Callable] = None,
                           on_cancel: Optional[Callable] = None,
                           confirm_text: str = "Confirm",
                           cancel_text: str = "Cancel",
                           danger_mode: bool = False) -> None:
    """Show a confirmation dialog using a SimpleDialog instance.
    
    Args:
        dialog: SimpleDialog instance
        title: Dialog title
        message: Dialog message
        on_confirm: Callback for confirm action
        on_cancel: Callback for cancel action
        confirm_text: Text for confirm button
        cancel_text: Text for cancel button
        danger_mode: Whether to use danger styling
    """
    dialog.show_confirmation(
        title=title,
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        danger_mode=danger_mode
    )


def show_info_dialog(dialog: SimpleDialog,
                    title: str,
                    message: str,
                    on_ok: Optional[Callable] = None,
                    ok_text: str = "OK",
                    info_type: str = "info") -> None:
    """Show an info dialog using a SimpleDialog instance.
    
    Args:
        dialog: SimpleDialog instance
        title: Dialog title
        message: Dialog message
        on_ok: Callback for OK action
        ok_text: Text for OK button
        info_type: Type of info dialog (info, success, warning, error)
    """
    dialog.show_info(
        title=title,
        message=message,
        on_ok=on_ok,
        ok_text=ok_text,
        info_type=info_type
    )


def show_success_dialog(dialog: SimpleDialog,
                       title: str,
                       message: str,
                       on_ok: Optional[Callable] = None,
                       ok_text: str = "OK") -> None:
    """Show a success dialog."""
    show_info_dialog(dialog, title, message, on_ok, ok_text, "success")


def show_warning_dialog(dialog: SimpleDialog,
                       title: str,
                       message: str,
                       on_ok: Optional[Callable] = None,
                       ok_text: str = "OK") -> None:
    """Show a warning dialog."""
    show_info_dialog(dialog, title, message, on_ok, ok_text, "warning")


def show_error_dialog(dialog: SimpleDialog,
                     title: str,
                     message: str,
                     on_ok: Optional[Callable] = None,
                     ok_text: str = "OK") -> None:
    """Show an error dialog."""
    show_info_dialog(dialog, title, message, on_ok, ok_text, "error")