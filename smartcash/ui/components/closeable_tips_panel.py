"""
Closeable Tips Panel Component

A reusable tips panel component with modern styling, flexible content, and close button.
"""

import ipywidgets as widgets
from typing import List, Dict, Optional, Union, Callable
import uuid

from smartcash.ui.components.tips_panel import create_tips_panel


def create_closeable_tips_panel(
    title: str = "💡 Tips & Requirements",
    tips: Optional[List[Union[str, List[str]]]] = None,
    gradient_start: str = "#e3f2fd",
    gradient_end: str = "#f3e5f5",
    border_color: str = "#2196f3",
    title_color: str = "#1976d2",
    text_color: str = "#424242",
    columns: int = 2,
    margin: str = "20px 0",
    on_close: Optional[Callable] = None,
    initially_visible: bool = True
) -> Dict[str, widgets.Widget]:
    """
    Create a modern tips panel with gradient background, flexible content, and close button.
    
    Args:
        title: Panel title with emoji
        tips: List of tips. Can be strings or lists of strings for multi-column layout
        gradient_start: Start color of the gradient background
        gradient_end: End color of the gradient background
        border_color: Left border color
        title_color: Title text color
        text_color: Tips text color
        columns: Number of columns for tips layout (1-4)
        margin: CSS margin for the panel
        on_close: Optional callback function to execute when panel is closed
        initially_visible: Whether the panel is initially visible
        
    Returns:
        Dictionary containing 'container' and 'set_visible' function
    """
    # Create unique ID for this panel
    panel_id = f"tips-panel-{uuid.uuid4().hex[:8]}"
    
    # Create the tips panel
    tips_panel = create_tips_panel(
        title=title,
        tips=tips,
        gradient_start=gradient_start,
        gradient_end=gradient_end,
        border_color=border_color,
        title_color=title_color,
        text_color=text_color,
        columns=columns,
        margin="0"  # We'll handle margin in the container
    )
    
    # Create close button
    close_button = widgets.Button(
        description='',
        icon='times',
        button_style='',
        tooltip='Close tips',
        layout=widgets.Layout(
            width='28px',
            height='28px',
            padding='0px',
            margin='0px'
        )
    )
    close_button.add_class('tips-close-button')
    
    # Add custom styling for the close button
    style_html = widgets.HTML(
        value=f"""
        <style>
            .tips-close-button {{
                position: absolute;
                top: 8px;
                right: 8px;
                opacity: 0.6;
                transition: opacity 0.2s;
            }}
            .tips-close-button:hover {{
                opacity: 1;
            }}
        </style>
        """
    )
    
    # Create container with relative positioning for absolute positioning of close button
    container_box = widgets.Box(
        [tips_panel, close_button, style_html],
        layout=widgets.Layout(
            width='100%',
            margin=margin,
            position='relative'
        )
    )
    container_box.add_class(panel_id)
    
    # Set initial visibility
    container_box.layout.display = 'flex' if initially_visible else 'none'
    
    # Define function to set visibility
    def set_visible(visible: bool):
        container_box.layout.display = 'flex' if visible else 'none'
    
    # Handle close button click
    def on_close_button_click(b):
        set_visible(False)
        if on_close:
            on_close()
    
    close_button.on_click(on_close_button_click)
    
    return {
        'container': container_box,
        'set_visible': set_visible,
        'panel_id': panel_id
    }
