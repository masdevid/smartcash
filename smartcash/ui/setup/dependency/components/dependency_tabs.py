"""
File: smartcash/ui/setup/dependency/components/dependency_tabs.py
Description: Enhanced tab container for package categories and custom packages with modern design
"""

import ipywidgets as widgets
from typing import Dict, Any

from .package_categories_tab import create_package_categories_tab
from .custom_packages_tab import create_custom_packages_tab

def create_dependency_tabs(config: Dict[str, Any], logger=None) -> widgets.Tab:
    """Create enhanced tabs for dependency management with full-width mx-auto styling."""
    
    # Create tabs with updated signatures
    package_tab = create_package_categories_tab(config, logger)
    custom_tab = create_custom_packages_tab(config, logger)
    
    # Create tab widget with full width and enhanced styling
    tabs = widgets.Tab(
        children=[package_tab, custom_tab],
        layout=widgets.Layout(
            width='100%',
            max_width='1200px',  # Set maximum width for better readability
            min_height='500px',
            margin='0 auto',  # Center the tabs (mx-auto equivalent)
            display='flex',
            flex_flow='column',
            align_items='stretch',  # Ensure children stretch to full width
            overflow_x='hidden'  # Prevent horizontal scrolling
        )
    )
    
    # Apply custom CSS to make tabs full width
    style = """
    <style>
        .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
            flex: 1 1 auto;
            justify-content: center;
        }
        .jupyter-widgets.widget-tab {
            width: 100% !important;
        }
    </style>
    """
    style_widget = widgets.HTML(value=style)
    
    # Set tab titles with emojis for better visual identification
    tabs.set_title(0, "📦 Package Categories")
    tabs.set_title(1, "🛠️ Custom Packages")
    
    # Store references for external access
    tabs.package_tab = package_tab
    tabs.custom_tab = custom_tab
    tabs.style_widget = style_widget
    
    # Create a container to hold both the style and tabs
    container = widgets.VBox([
        style_widget,
        tabs
    ], layout=widgets.Layout(width='100%', margin='0 auto'))
    
    # Make the container look like a Tab widget by exposing the same interface
    container.children = tabs.children
    container.package_tab = tabs.package_tab
    container.custom_tab = tabs.custom_tab
    container.set_title = tabs.set_title
    container.selected_index = tabs.selected_index
    
    return container