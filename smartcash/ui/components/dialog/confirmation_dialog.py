"""
Confirmation dialog component for displaying modal dialogs with confirmation actions.

This module provides a backward-compatible interface to the new SimpleDialog implementation.
New code should use the SimpleDialog class directly from simple_dialog.py.
"""

import warnings
from typing import Dict, Any, Callable, Optional
from IPython.display import display, clear_output, HTML

from smartcash.ui.components.dialog.simple_dialog import (
    SimpleDialog,
    show_confirmation_dialog as simple_show_confirmation_dialog,
    show_info_dialog as simple_show_info_dialog,
    show_success_dialog as simple_show_success_dialog,
    show_warning_dialog as simple_show_warning_dialog,
    show_error_dialog as simple_show_error_dialog
)


class ConfirmationDialog(SimpleDialog):
    """A confirmation dialog component that wraps SimpleDialog for backward compatibility.
    
    This class is maintained for backward compatibility. New code should use SimpleDialog directly.
    """
    
    def __init__(self, 
                 component_name: str = "confirmation_dialog",
                 **kwargs):
        """Initialize the confirmation dialog.
        
        Args:
            component_name: Unique name for this component
            **kwargs: Additional arguments to pass to SimpleDialog
        """
        warnings.warn(
            "ConfirmationDialog is deprecated. Use SimpleDialog from simple_dialog.py instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(component_name, **kwargs)
    
    def show(self, 
             title: str,
             message: str,
             on_confirm: Optional[Callable] = None,
             on_cancel: Optional[Callable] = None,
             confirm_text: str = "Konfirmasi",
             cancel_text: str = "Batal",
             danger_mode: bool = False) -> None:
        """Show the confirmation dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_confirm: Callback for confirm action
            on_cancel: Callback for cancel action
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            danger_mode: Whether to use danger styling
        """
        # Use the parent SimpleDialog method instead of duplicating HTML
        self.show_confirmation(
            title=title,
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode
        )
    
    def show_info(self,
                 title: str,
                 message: str,
                 on_ok: Optional[Callable] = None,
                 ok_text: str = "OK") -> None:
        """Show an info dialog with a single OK button.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback for OK button
            ok_text: Text for OK button
        """
        self.show(
            title=title,
            message=message,
            on_confirm=on_ok,
            on_cancel=None,
            confirm_text=ok_text,
            cancel_text="",
            danger_mode=False
        )
    
    def _show_dialog(self, html_content: str) -> None:
        """Display the dialog with the given HTML content."""
        if not self._initialized:
            self.initialize()
        
        container = self._ui_components['container']
        
        # Always clear first to ensure clean state
        with container:
            clear_output(wait=True)
        
        # Update container styles
        container.layout.display = 'flex'
        container.layout.visibility = 'visible'
        container.layout.height = 'auto'
        container.layout.min_height = '200px'
        container.layout.max_height = '300px'
        container.layout.padding = '10px 15px'
        container.layout.margin = '5px 0'
        container.layout.overflow = 'hidden'
        container.layout.flex = '0 0 auto'
        
        # Display the dialog
        with container:
            display(HTML(html_content))
        
        self._is_visible = True
    
    def hide(self) -> None:
        """Hide the dialog."""
        if not self._initialized:
            return
        
        container = self._ui_components.get('container')
        if container:
            # Clear the output first
            with container:
                clear_output(wait=True)
            
            # Reset container styles to initial state
            container.layout.display = 'none'
            container.layout.visibility = 'hidden'
            container.layout.height = None
            container.layout.min_height = '50px'
            container.layout.max_height = '500px'
            container.layout.padding = '10px'
            container.layout.margin = '10px 0'
            container.layout.overflow = 'auto'
            container.layout.border = '1px solid #e0e0e0'
            container.layout.border_radius = '4px'
            container.layout.flex = '1 1 auto'
        
        # Clean up state
        self._is_visible = False
        self._callbacks = {}
    
    def is_visible(self) -> bool:
        """Check if the dialog is currently visible."""
        if not self._initialized:
            return False
        
        container = self._ui_components.get('container')
        if container and hasattr(container, 'layout'):
            return container.layout.display != 'none'
        
        return self._is_visible


# Backward compatibility functions
def create_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Legacy function to create a confirmation area.
    
    This is a compatibility function that creates a SimpleDialog instance
    and stores it in the provided ui_components dictionary.
    
    Args:
        ui_components: Dictionary to store UI components
    """
    warnings.warn(
        "create_confirmation_area is deprecated. Use SimpleDialog directly instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if 'confirmation_dialog' not in ui_components:
        ui_components['confirmation_dialog'] = ConfirmationDialog()


def show_confirmation_dialog(ui_components: Dict[str, Any],
                           title: str,
                           message: str,
                           on_confirm: Optional[Callable] = None,
                           on_cancel: Optional[Callable] = None,
                           confirm_text: str = "Konfirmasi",
                           cancel_text: str = "Batal",
                           danger_mode: bool = False) -> None:
    """Legacy function to show a confirmation dialog."""
    warnings.warn(
        "show_confirmation_dialog is deprecated. Use SimpleDialog.show_confirmation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if 'confirmation_dialog' not in ui_components:
        create_confirmation_area(ui_components)
    
    dialog = ui_components['confirmation_dialog']
    dialog.show(
        title=title,
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        danger_mode=danger_mode
    )


def show_info_dialog(ui_components: Dict[str, Any],
                   title: str,
                   message: str,
                   on_ok: Optional[Callable] = None,
                   ok_text: str = "OK") -> None:
    """Legacy function to show an info dialog."""
    warnings.warn(
        "show_info_dialog is deprecated. Use SimpleDialog.show_info() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if 'confirmation_dialog' not in ui_components:
        create_confirmation_area(ui_components)
    
    dialog = ui_components['confirmation_dialog']
    dialog.show_info(
        title=title,
        message=message,
        on_ok=on_ok,
        ok_text=ok_text
    )


def clear_dialog_area(ui_components: Dict[str, Any]) -> None:
    """Legacy function to clear the dialog area."""
    warnings.warn(
        "clear_dialog_area is deprecated. Use SimpleDialog.hide() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if 'confirmation_dialog' in ui_components:
        dialog = ui_components['confirmation_dialog']
        dialog.hide()

    if 'confirmation_area' in ui_components:
        confirmation_area = ui_components['confirmation_area']
        if hasattr(confirmation_area, 'layout') and hasattr(confirmation_area.layout, 'visibility'):
            confirmation_area.layout.visibility = 'hidden'

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Legacy function to check if dialog is visible."""
    if 'confirmation_dialog' in ui_components:
        return ui_components['confirmation_dialog'].is_visible()
    elif 'confirmation_area' in ui_components:
        confirmation_area = ui_components['confirmation_area']
        return (hasattr(confirmation_area, 'layout') and 
                confirmation_area.layout.display != 'none')
    return False
