"""
File: smartcash/ui/components/action_container.py
Deskripsi: Simplified action container with flex column layout that reuses existing components
"""
from typing import Dict, Any, Optional, List, Literal, Callable
import ipywidgets as widgets

# Import existing components
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.dialog.confirmation_dialog import (
    create_confirmation_area,
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible
)

def create_action_container(
    buttons: List[Dict[str, Any]],
    title: str = "🚀 Operations",
    alignment: Literal['left', 'center', 'right'] = 'left',
    container_margin: str = "10px 0"
) -> Dict[str, Any]:
    """
    Create a simplified action container with flex column layout that reuses existing components

    Args:
        buttons: List of button configurations, each a dictionary with:
            - button_id: Unique identifier for the button (required)
            - text: Button label text (required)
            - icon: Button icon name (optional)
            - style: Button style ('primary', 'success', 'info', 'warning', 'danger', '') (optional)
            - tooltip: Button tooltip text (optional)
            - order: Display order (lower numbers first) (optional)
        title: Section title (default: "🚀 Operations")
        alignment: Button container alignment ('left', 'center', 'right') (default: 'left')
        container_margin: Margin around the container (default: "10px 0")

    Returns:
        Dictionary containing:
            - container: The main container widget
            - buttons: Dictionary of button widgets by button_id
            - dialog_area: The dialog area widget
            - show_dialog: Function to show a dialog
            - show_info: Function to show an info dialog
            - clear_dialog: Function to clear the dialog
            - is_dialog_visible: Function to check if dialog is visible

    Examples:
        >>> action_container = create_action_container([
        ...     {'button_id': 'process', 'text': '🚀 Process', 'icon': 'play', 'style': 'primary'},
        ...     {'button_id': 'cancel', 'text': '❌ Cancel', 'style': 'danger'}
        ... ], title="💼 Process Data")
        >>> display(action_container['container'])
        >>>
        >>> # Show a dialog
        >>> action_container['show_dialog'](
        ...     title="Confirm Process",
        ...     message="Are you sure you want to process the data?",
        ...     on_confirm=lambda: print("Processing..."),
        ...     on_cancel=lambda: print("Cancelled")
        ... )
        >>>
        >>> # Access a button
        >>> process_btn = action_container['buttons']['process']
        >>> process_btn.on_click(lambda b: print("Button clicked"))
    """
    # Create the title HTML
    title_widget = widgets.HTML(
        f"<div style='font-weight:bold;color:#28a745;margin-bottom:8px;'>{title}</div>"
    )

    # Create action buttons using the existing component
    action_buttons = create_action_buttons(
        buttons=buttons,
        alignment=alignment
    )

    # Create a dictionary to store UI components
    ui_components = {}

    # Create dialog area using the existing component
    dialog_area = create_confirmation_area(ui_components)

    
    # Create the container components list
    container_components = [title_widget, action_buttons['container'], dialog_area]
    
    # Create the container with flex column layout
    container = widgets.VBox(
        container_components,
        layout=widgets.Layout(
            display='flex',
            flex_direction='column',
            width='100%',
            margin=container_margin,
            align_items='stretch'
        )
    )
    
    # Create wrapper functions for dialog operations
    def show_dialog(title, message, on_confirm=None, on_cancel=None, confirm_text="Konfirmasi", cancel_text="Batal", danger_mode=False):
        return show_confirmation_dialog(
            ui_components=ui_components,
            title=title,
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode
        )
    
    def show_info(title, message, on_ok=None, ok_text="OK"):
        return show_info_dialog(
            ui_components=ui_components,
            title=title,
            message=message,
            on_ok=on_ok,
            ok_text=ok_text
        )
    
    def clear_dialog():
        return clear_dialog_area(ui_components)
    
    def check_dialog_visible():
        return is_dialog_visible(ui_components)
    
    # Return result with container and references
    result = {
        'container': container,
        'buttons': action_buttons['buttons'],
        'dialog_area': dialog_area,
        'show_dialog': show_dialog,
        'show_info': show_info,
        'clear_dialog': clear_dialog,
        'is_dialog_visible': check_dialog_visible
    }
    
    return result
