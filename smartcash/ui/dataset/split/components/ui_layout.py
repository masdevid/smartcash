"""
File: smartcash/ui/dataset/split/components/ui_layout.py

Layout components for dataset split configuration.
Uses shared components from the UI components directory.
"""

from typing import Dict, Any, List
import ipywidgets as widgets

# Shared components

# Local imports
from smartcash.ui.dataset.split.components.ui_form import (
    create_ratio_section,
    create_path_section
)

# Constants
STYLES = {
    'container': {
        'width': '100%',
        'margin': '10px 0',
        'padding': '15px',
        'border': '1px solid #e0e0e0',
        'border_radius': '4px',
        'background': '#fff'
    }
}

def create_responsive_two_column(left: widgets.Widget, right: widgets.Widget) -> widgets.HBox:
    """Create a responsive two-column layout.
    
    Args:
        left: Left column widget
        right: Right column widget
        
    Returns:
        widgets.HBox: Two-column layout container
    """
    return widgets.HBox(
        [left, right],
        layout=widgets.Layout(
            width='100%',
            justify_content='space-between',
            flex_flow='row wrap'
        )
    )

def create_split_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create the main layout for dataset split configuration using a two-column grid.
    
    Args:
        form_components: Dictionary of form components
        
    Returns:
        Dictionary containing the container and content area
    """
    # Create sections
    ratio_section = create_ratio_section(form_components)
    path_section = create_path_section(form_components)
    
    # Create a two-column grid layout
    grid = widgets.GridBox(
        children=[ratio_section, path_section],
        layout=widgets.Layout(
            width='100%',
            grid_template_columns='1fr 1fr',  # Two equal columns
            grid_gap='20px',
            margin='0 0 20px 0'
        )
    )
    
    # Get save/reset buttons if they exist
    save_reset_buttons = form_components.get('save_reset_buttons', {})
    save_reset_container = save_reset_buttons.get('container', widgets.HBox())
    
    # Create form container with the grid and save/reset buttons
    form_container = widgets.VBox(
        [
            grid,
            save_reset_container
        ],
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            background='#fff',
            margin_top='10px'
        )
    )
    
    # Add the container to form_components for easy access
    if 'save_reset_buttons' in form_components:
        form_components['save_reset_buttons']['container'] = save_reset_container
    
    return {
        'container': form_container,
        'content': form_container
    }
    
