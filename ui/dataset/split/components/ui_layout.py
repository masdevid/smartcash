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
    """Create the main layout for dataset split configuration.
    
    Args:
        form_components: Dictionary of form components
        
    Returns:
        Dictionary containing the main container and content area
    """
    # Create header
    header = create_header(
        "Dataset Split Configuration",
        "Configure how to split your dataset into training, validation, and test sets"
    )
    
    # Create sections
    ratio_section = create_ratio_section(form_components)
    path_section = create_path_section(form_components)
    
    # Create form container with responsive layout
    form_container = widgets.VBox(
        [
            create_responsive_two_column(ratio_section, path_section),
            form_components.get('save_reset_container', widgets.HTML(''))
        ],
        layout=widgets.Layout(**STYLES['container'])
    )
    
    # Create main container
    main_container = widgets.VBox(
        [header, form_container],
        layout=widgets.Layout(width='100%')
    )
    
    return {
        'container': main_container,
        'content': form_container
    }
    
