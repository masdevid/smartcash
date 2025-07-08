"""
File: smartcash/ui/setup/dependency/components/dependency_tabs.py
Description: Enhanced tab container for package categories and custom packages with modern design
"""

import ipywidgets as widgets
from typing import Dict, Any

from .package_categories_tab import create_package_categories_tab
from .custom_packages_tab import create_custom_packages_tab

def create_dependency_tabs(config: Dict[str, Any], logger=None) -> widgets.Tab:
    """Create enhanced tabs for dependency management with responsive design."""
    
    # Create tabs with updated signatures
    package_tab = create_package_categories_tab(config, logger)
    custom_tab = create_custom_packages_tab(config, logger)
    
    # Create tab widget with enhanced styling
    tabs = widgets.Tab(
        children=[package_tab, custom_tab],
        layout=widgets.Layout(
            width='100%',
            min_height='500px',  # Reduced min height for better responsiveness
            margin='0'  # Remove margin for tighter layout
        )
    )
    
    # Set tab titles with emojis for better visual identification
    tabs.set_title(0, "📦 Package Categories")
    tabs.set_title(1, "🛠️ Custom Packages")
    
    # Store references for external access
    tabs.package_tab = package_tab
    tabs.custom_tab = custom_tab
    
    return tabs