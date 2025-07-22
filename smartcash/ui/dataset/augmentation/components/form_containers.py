"""
File: smartcash/ui/dataset/augmentation/components/containers.py
Description: Container components for the augmentation UI module.
"""

# Standard library imports
from typing import Dict, Any

# Third-party imports
import ipywidgets as widgets

# Local imports - removed SECTION_STYLES dependency


def create_form_container(
    content: widgets.Widget, 
    title: str, 
    theme: str = 'basic', 
    width: str = '48%'
) -> widgets.VBox:
    """Create a styled container with consistent theming.
    
    Args:
        content: Widget content to be wrapped in the container
        title: Title to display in the container header
        theme: Visual theme to apply (default: 'basic')
        width: Width of the container (default: '48%')
        
    Returns:
        A styled VBox container with the specified content and theming
    """
    # Use original color scheme from constants
    # Use simple default styles instead of SECTION_STYLES
    border_color = '#e0e0e0'
    bg_color = '#f8f9fa'
    
    header_html = f"""
    <div style="padding: 8px 12px; margin-bottom: 8px;
                background: linear-gradient(145deg, {bg_color} 0%, rgba(255,255,255,0.9) 100%);
                border-radius: 8px; border-left: 4px solid {border_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h5 style="color: #333; margin: 0; font-size: 14px; font-weight: 600;">
            {title}
        </h5>
    </div>
    """
    
    return widgets.VBox([
        widgets.HTML(header_html),
        content
    ], layout=widgets.Layout(
        width=width,
        margin='5px',
        padding='10px',
        border=f'1px solid {border_color}',
        border_radius='8px',
        background_color='rgba(255,255,255,0.8)',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        flex=f'1 1 {width.replace("%", "")}%' if '%' in width else '1 1 auto'
    ))
