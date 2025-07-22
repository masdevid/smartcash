"""
Info Panel Component

A reusable component for displaying information in a modern, styled panel.
"""

import ipywidgets as widgets
from typing import Dict, List, Optional, Union

def create_info_panel(
    title: str,
    items: List[Dict[str, str]],
    icon: str = "ℹ️",
    width: str = "48%",
    margin: str = "10px 1% 10px 0",
    border_color: str = "#e9ecef",
    title_color: str = "#2c3e50",
    text_color: str = "#555",
) -> widgets.HTML:
    """
    Create a modern info panel with a title and list of key-value items.
    
    Args:
        title: Panel title
        items: List of dicts with 'key' and 'value_id' (e.g., [{'key': 'Platform', 'value_id': 'platform-info'}])
        icon: Emoji icon to display before the title
        width: CSS width of the panel
        margin: CSS margin
        border_color: Border color
        title_color: Title text color
        text_color: Content text color
        
    Returns:
        IPython HTML widget containing the info panel
    """
    # Generate items HTML
    items_html = "\n".join(
        f'<div style="margin: 8px 0; display: flex; align-items: baseline;">'
        f'<strong style="min-width: 100px;">{item["key"]}:</strong> '
        f'<span id="{item["value_id"]}" style="flex: 1;">Memuat...</span>'
        '</div>'
        for item in items
    )
    
    # Generate the complete HTML
    html = f"""
    <div style="
        padding: 15px;
        border: 1px solid {border_color};
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        height: 100%;
        box-sizing: border-box;
        transition: all 0.3s ease;
    ">
        <h4 style="
            color: {title_color};
            margin: 0 0 12px 0;
            padding-bottom: 8px;
            font-size: 16px;
            font-weight: 600;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="font-size: 1.2em;">{icon}</span>
            {title}
        </h4>
        <div style="
            color: {text_color};
            font-size: 13px;
            line-height: 1.5;
        ">
            {items_html}
        </div>
    </div>
    """
    
    return widgets.HTML(
        value=html,
        layout=widgets.Layout(
            width=width,
            margin=margin,
            display='flex',
            flex_direction='column'
        )
    )

def create_dual_info_panels(
    left_panel: Dict[str, any],
    right_panel: Dict[str, any],
    gap: str = '2%',
    min_width: str = '250px',
    **layout_kwargs
) -> widgets.HTML:
    """
    Create two info panels side by side using CSS flexbox.
    
    Args:
        left_panel: Configuration for the left panel (kwargs for create_info_panel)
        right_panel: Configuration for the right panel (kwargs for create_info_panel)
        gap: Space between panels (CSS value)
        min_width: Minimum width of each panel (CSS value)
        **layout_kwargs: Additional layout properties for the container
        
    Returns:
        HTML widget containing the two panels in a flex container
    """
    # Create panels with consistent styling
    left = create_info_panel(**{
        **left_panel,
        'width': f'calc(50% - {gap}/2)',
        'margin': '0',
    })
    
    right = create_info_panel(**{
        **right_panel,
        'width': f'calc(50% - {gap}/2)',
        'margin': '0',
    })
    
    # Create flex container with panels
    html = f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: {gap};
        width: 100%;
        margin: 20px 0;
        {layout_kwargs.get('style', '')}
    ">
        <div style="
            flex: 1 1 {min_width};
            min-width: {min_width};
            max-width: 100%;
        ">
            {left.value}
        </div>
        <div style="
            flex: 1 1 {min_width};
            min-width: {min_width};
            max-width: 100%;
        ">
            {right.value}
        </div>
    </div>
    """
    
    return widgets.HTML(html)
