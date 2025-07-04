"""
File: smartcash/ui/setup/dependency/components/dependency_tabs.py
Deskripsi: Tab container untuk package categories dan custom packages
"""

import ipywidgets as widgets
from typing import Dict, Any

from .package_categories_tab import create_package_categories_tab
from .custom_packages_tab import create_custom_packages_tab

def create_dependency_tabs(config: Dict[str, Any], logger) -> widgets.Tab:
    """Create tabs untuk dependency management"""
    
    # Create tabs
    package_tab = create_package_categories_tab(config, logger)
    custom_tab = create_custom_packages_tab(config, logger)
    
    # Create tab widget
    tabs = widgets.Tab(
        children=[package_tab, custom_tab],
        layout=widgets.Layout(
            width='100%',
            min_height='600px',
            margin='10px 0'
        )
    )
    
    tabs.set_title(0, "📦 Package Categories")
    tabs.set_title(1, "🛠️ Custom Packages")
    
    return tabs