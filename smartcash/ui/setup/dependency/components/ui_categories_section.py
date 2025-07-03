"""
File: smartcash/ui/setup/dependency/components/ui_categories_section.py
Description: UI component for displaying and selecting package categories
"""

from typing import Dict, List, Any, Optional, Callable
import ipywidgets as widgets

from smartcash.ui.components.card import create_card
from smartcash.ui.components import create_form_container


class PackageItem(widgets.HBox):
    """A reusable package item widget with checkbox and description."""
    
    def __init__(
        self,
        package: Dict[str, Any],
        is_checked: bool = False,
        on_select: Optional[Callable[[str, bool], None]] = None
    ):
        """Initialize a package item widget.
        
        Args:
            package: Dictionary containing package details
            is_checked: Whether the package is selected
            on_select: Callback function when selection changes
        """
        self.package = package
        self.package_key = package.get('key', '')
        self.package_name = package.get('name', self.package_key)
        self.package_description = package.get('description', '')
        self.package_version = package.get('version', '')
        
        # Create checkbox
        self.checkbox = widgets.Checkbox(
            value=is_checked,
            description='',
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        
        # Create package info
        package_info = widgets.HTML(
            f"""
            <div style='margin-left: 5px;'>
                <div style='font-weight: 600; font-size: 0.9em;'>{self.package_name}</div>
                <div style='color: #666; font-size: 0.8em;'>{self.package_description}</div>
                <div style='color: #888; font-size: 0.75em;'>Version: {self.package_version}</div>
            </div>
            """
        )
        
        # Register callback
        def on_checkbox_change(change):
            if on_select and change.get('type') == 'change' and change.get('name') == 'value':
                on_select(self.package_key, change.get('new'))
                
        self.checkbox.observe(on_checkbox_change)
        
        # Initialize the HBox
        super().__init__(
            [self.checkbox, package_info],
            layout=widgets.Layout(
                margin='5px 0',
                padding='5px 10px',
                border='1px solid #e0e0e0',
                border_radius='4px',
                width='100%',
                align_items='center'
            )
        )
        
        # Add hover effect
        self.add_class('package-item')


class CategoryCard(widgets.VBox):
    """A reusable category card widget with packages."""
    
    def __init__(
        self,
        category: Dict[str, Any],
        selected_packages: List[str],
        on_package_select: Optional[Callable[[str, bool], None]] = None
    ):
        """Initialize a category card widget.
        
        Args:
            category: Dictionary containing category details
            selected_packages: List of selected package keys
            on_package_select: Callback function when package selection changes
        """
        # Get category details
        self.name = category.get('name', 'Unnamed Category')
        self.description = category.get('description', '')
        self.icon = category.get('icon', '📦')
        self.packages = category.get('packages', [])
        
        # Handle empty packages case
        if not self.packages:
            empty_message = widgets.HTML(
                f"<p style='padding: 8px; color: #666;'>No packages in {self.name} category</p>"
            )
            super().__init__([empty_message])
            return
            
        # Create package items
        self.package_items = []
        for package in self.packages:
            is_checked = package.get('key', '') in selected_packages
            package_item = PackageItem(
                package=package,
                is_checked=is_checked,
                on_select=on_package_select
            )
            self.package_items.append(package_item)
        
        # Create package list
        self.package_list = widgets.VBox(
            self.package_items,
            layout=widgets.Layout(
                width='100%',
                margin='8px 0',
                max_height='200px',
                overflow_y='auto'
            )
        )
        
        # Create card widget
        self.card_widget = create_card(
            title=self.name,
            value=f"{len(self.packages)} packages",
            icon=self.icon,
            color="#4285F4",
            description=self.description,
            extra_content=self.package_list
        )
        
        # Initialize the VBox
        super().__init__(
            [self.card_widget],
            layout=widgets.Layout(
                margin='10px 0',
                width='100%'
            )
        )


def create_categories_section(
    config: Dict[str, Any],
    on_package_select: Optional[Callable[[str, bool], None]] = None
) -> widgets.Widget:
    """Create a section with package categories.
    
    Args:
        config: Configuration dictionary with categories and selected packages
        on_package_select: Callback function when package selection changes
        
    Returns:
        Widget containing the categories section
    """
    # Get categories and selected packages from config
    categories = config.get('categories', [])
    selected_packages = config.get('selected_packages', [])
    
    # Create header
    header = widgets.HTML(
        "<h3 style='margin: 0 0 12px 0;'>Package Categories</h3>"
    )
    
    # Create category cards
    category_widgets = []
    for category in categories:
        category_card = CategoryCard(
            category=category,
            selected_packages=selected_packages,
            on_package_select=on_package_select
        )
        category_widgets.append(category_card)
    
    # Create grid layout for categories
    grid_layout = widgets.Layout(
        display='grid',
        grid_template_columns='repeat(auto-fill, minmax(300px, 1fr))',
        grid_gap='16px',
        width='100%'
    )
    
    # Create grid container
    grid_container = widgets.Box(
        category_widgets,
        layout=grid_layout
    )
    
    # Create container for all categories with header
    container = widgets.VBox(
        [header, grid_container],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    return container
