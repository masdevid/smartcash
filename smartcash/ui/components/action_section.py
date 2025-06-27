"""
File: smartcash/ui/components/action_section.py
Deskripsi: Shared component for action sections with consistent styling and layout
"""
from typing import Dict, Any, Optional, Union
import ipywidgets as widgets

def create_action_section(
    action_buttons: Union[Dict[str, Any], widgets.Widget],
    confirmation_area: widgets.Widget,
    title: str = "🚀 Operations",
    status_label: str = "📋 Status & Konfirmasi:",
    show_status: bool = True
) -> widgets.VBox:
    """
    Create a standardized action section with buttons and confirmation area
    
    Args:
        action_buttons: Either a dictionary containing the action buttons container or a widget
        confirmation_area: Widget for confirmation dialogs
        title: Section title (default: "🚀 Operations")
        status_label: Label for the status section (default: "📋 Status & Konfirmasi:")
        show_status: Whether to show the status section (default: True)
        
    Returns:
        widgets.VBox: A container with the action section
    """
    # Create the title HTML
    title_widget = widgets.HTML(
        f"<div style='font-weight:bold;color:#28a745;margin-bottom:8px;'>{title}</div>"
    )
    
    # Create the status section if enabled
    status_section = []
    if show_status:
        status_section = [
            widgets.HTML(
                f"<div style='margin:8px 0 4px 0;font-size:13px;color:#666;'>"
                f"<strong>{status_label}</strong></div>"
            ),
            widgets.Box(
                [confirmation_area],
                layout=widgets.Layout(
                    display='flex',
                    flex_flow='row wrap',
                    justify_content='space-between',
                    align_items='center',
                    width='100%',
                    margin='0',
                    padding='0'
                )
            )
        ]
    
    # Get the buttons container from either dict or widget
    buttons_container = action_buttons['container'] if isinstance(action_buttons, dict) else action_buttons
    
    # Create the action section
    action_section = widgets.VBox(
        [title_widget, buttons_container] + status_section,
        layout=widgets.Layout(
            display='flex',
            flex_direction='column',
            width='100%',
            margin='10px 0',
            padding='12px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#f9f9f9',
            overflow='hidden'
        )
    )
    
    return action_section
