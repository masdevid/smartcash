"""
File: smartcash/ui/setup/dependency/components/ui_custom_packages_section.py
Description: UI component for adding and managing custom packages
"""

from typing import Dict, List, Any, Optional, Callable
import ipywidgets as widgets

from smartcash.ui.components.card import create_card


class CustomPackageItem(widgets.HBox):
    """A reusable custom package item widget with remove button."""
    
    def __init__(
        self,
        package_spec: str,
        on_remove: Optional[Callable[[str], None]] = None
    ):
        """Initialize a custom package item widget.
        
        Args:
            package_spec: Package specification string (e.g., "numpy>=1.20.0")
            on_remove: Callback function when package is removed
        """
        self.package_spec = package_spec
        
        # Create package label
        package_label = widgets.HTML(
            f"""
            <div style='
                font-family: monospace;
                padding: 2px 8px;
                background-color: #f1f3f4;
                border-radius: 4px;
                font-size: 0.9em;
            '>
                {package_spec}
            </div>
            """
        )
        
        # Create remove button
        remove_button = widgets.Button(
            description='',
            icon='trash',
            button_style='danger',
            layout=widgets.Layout(width='32px', height='32px')
        )
        
        # Register callback
        def on_remove_click(b):
            if on_remove:
                on_remove(self.package_spec)
                
        remove_button.on_click(on_remove_click)
        
        # Add spacer for layout
        spacer = widgets.Box(layout=widgets.Layout(flex='1 1 auto'))
        
        # Initialize the HBox
        super().__init__(
            [package_label, spacer, remove_button],
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
        self.add_class('custom-package-item')


class CustomPackagesList(widgets.VBox):
    """A list of custom packages with add/remove functionality."""
    
    def __init__(
        self,
        initial_packages: List[str] = None,
        on_packages_change: Optional[Callable[[List[str]], None]] = None
    ):
        """Initialize a custom packages list widget.
        
        Args:
            initial_packages: List of initial package specifications
            on_packages_change: Callback function when package list changes
        """
        self.packages = initial_packages or []
        self.on_packages_change = on_packages_change
        
        # Create input field for package names with validation
        self.package_input = widgets.Text(
            placeholder='Example: numpy>=1.20.0, pandas<2.0.0',
            description='Package:',
            layout=widgets.Layout(width='70%'),
            style={'description_width': 'initial'}
        )
        
        # Create add button
        self.add_button = widgets.Button(
            description='Add',
            button_style='primary',
            icon='plus',
            layout=widgets.Layout(width='100px')
        )
        
        # Create input row
        input_row = widgets.HBox(
            [self.package_input, self.add_button],
            layout=widgets.Layout(
                width='100%',
                margin='10px 0'
            )
        )
        
        # Create package list container
        self.package_list = widgets.VBox(
            [],
            layout=widgets.Layout(
                width='100%',
                max_height='200px',
                overflow_y='auto'
            )
        )
        
        # Create empty state message
        self.empty_message = widgets.HTML(
            """
            <div style='
                color: #666;
                text-align: center;
                padding: 20px;
                font-style: italic;
            '>
                No custom packages added yet.
                Add packages using the input field above.
            </div>
            """
        )
        
        # Register callbacks
        def on_add_click(b):
            self._add_package()
            
        self.add_button.on_click(on_add_click)
        
        def on_input_keypress(widget, event):
            if event.get('type') == 'keydown' and event.get('key') == 'Enter':
                self._add_package()
                
        self.package_input.on_keydown(on_input_keypress)
        
        # Initialize with any provided packages
        self._refresh_package_list()
        
        # Initialize the VBox
        super().__init__(
            [input_row, self.package_list],
            layout=widgets.Layout(
                width='100%',
                padding='10px'
            )
        )
    
    def _add_package(self):
        """Add a package from the input field."""
        package_spec = self.package_input.value.strip()
        if package_spec and package_spec not in self.packages:
            self.packages.append(package_spec)
            self._refresh_package_list()
            self.package_input.value = ''
            
            if self.on_packages_change:
                self.on_packages_change(self.packages)
    
    def _remove_package(self, package_spec: str):
        """Remove a package from the list."""
        if package_spec in self.packages:
            self.packages.remove(package_spec)
            self._refresh_package_list()
            
            if self.on_packages_change:
                self.on_packages_change(self.packages)
    
    def _refresh_package_list(self):
        """Refresh the package list display."""
        # Clear current list
        self.package_list.children = ()
        
        if not self.packages:
            self.package_list.children = (self.empty_message,)
            return
            
        # Create package items
        package_items = []
        for package_spec in self.packages:
            package_item = CustomPackageItem(
                package_spec=package_spec,
                on_remove=self._remove_package
            )
            package_items.append(package_item)
            
        self.package_list.children = tuple(package_items)


def create_custom_packages_section(
    initial_packages: List[str] = None,
    on_packages_change: Optional[Callable[[List[str]], None]] = None
) -> widgets.Widget:
    """Create a section for adding and managing custom packages.
    
    Args:
        initial_packages: List of initial package specifications
        on_packages_change: Callback function when package list changes
        
    Returns:
        Widget containing the custom packages section
    """
    # Create header
    header = widgets.HTML(
        "<h3 style='margin: 0 0 12px 0;'>Custom Packages</h3>"
    )
    
    # Create description
    description = widgets.HTML(
        """
        <p style='margin: 0 0 16px 0; color: #666;'>
            Add custom packages with specific version requirements.
            Use standard pip syntax (e.g., "numpy>=1.20.0", "pandas==1.5.3").
        </p>
        """
    )
    
    # Create custom packages list
    packages_list = CustomPackagesList(
        initial_packages=initial_packages,
        on_packages_change=on_packages_change
    )
    
    # Create container
    container = widgets.VBox(
        [header, description, packages_list],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    return container
