"""
File: smartcash/ui/setup/dependency/components/ui_package_selector.py
Deskripsi: Package selector utilities with improved spacing and justify alignment
"""

# Standard library imports
from typing import Dict, Any, Optional, List

# Third-party imports
import ipywidgets as widgets

# Local application imports
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.setup.dependency.utils.package_categories import get_package_categories

def create_package_selector_grid(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create package selector grid with improved spacing and justify alignment
    
    Args:
        config: Optional configuration dictionary. If provided, will be used to set initial values.
    
    Returns:
        Dict containing the container widget, checkboxes, and categories
    """
    # Get package categories from defaults
    package_categories = get_package_categories()
    
    # Get selected packages from config if provided
    selected_packages = set(config.get('selected_packages', [])) if config else set()
    
    checkboxes = {}
    category_widgets = []
    
    # Create category boxes with improved spacing
    for category in package_categories:
        category_widget = _create_category_widget_improved(category, selected_packages)
        checkboxes.update(category_widget['checkboxes'])
        category_widgets.append(category_widget['widget'])
    
    # Create main container with consistent spacing
    container = widgets.VBox(
        category_widgets,
        layout=widgets.Layout(
            width='100%',
            margin='0 0 20px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='5px',
            overflow_y='auto',
            max_height='500px'
        )
    )
    
    return {
        'container': container,
        'checkboxes': checkboxes,
        'categories': package_categories
    }

def _create_category_widget_improved(
    category: Dict[str, Any], 
    selected_packages: Optional[set] = None
) -> Dict[str, Any]:
    """Create single category widget with improved spacing and justify alignment
    
    Args:
        category: Category configuration dictionary
        selected_packages: Set of package keys that should be selected by default
        
    Returns:
        Dict containing the category widget and checkboxes
    """
    checkboxes = {}
    selected_packages = selected_packages or set()
    
    # Create checkboxes for each package in the category
    package_widgets = []
    for pkg in category['packages']:
        pkg_key = pkg['key']
        
        # Determine if package should be checked
        is_default = pkg.get('default', False)
        is_selected = pkg_key in selected_packages if selected_packages else is_default
        
        checkbox = widgets.Checkbox(
            value=is_selected,
            description=pkg['name'],
            disabled=not pkg.get('optional', True),
            indent=False,
            layout=widgets.Layout(
                width='auto',
                margin='2px 0',
                padding='4px 8px',
                border_radius='4px',
                border='1px solid #e0e0e0',
                background_color='#f9f9f9'
            )
        )
        
        # Add tooltip if description exists
        if 'description' in pkg:
            checkbox.add_class('has-tooltip')
            checkbox.tooltip = pkg['description']
        
        checkboxes[pkg_key] = checkbox
        
        # Create status indicator
        status = widgets.HTML(
            value='',
            placeholder='',
            description='',
            layout=widgets.Layout(
                width='24px',
                height='24px',
                margin='0 0 0 8px',
                padding='0',
                display='flex',
                align_items='center',
                justify_content='center'
            )
        )
        
        # Create row with checkbox and status
        row = widgets.HBox(
            [checkbox, status],
            layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                align_items='center',
                margin='2px 0',
                padding='2px 0'
            )
        )
        
        # Store status widget reference in checkbox for easy updates
        checkbox.status = status
        package_widgets.append(row)
    
    # Create category header
    header = widgets.HTML(
        value=f"<b>{category['name']}</b>" + 
              (f"<p style='margin: 4px 0 8px 0; color: #666; font-size: 0.9em;'>{category.get('description', '')}</p>" 
               if 'description' in category else ''),
        layout=widgets.Layout(
            width='100%',
            margin='0 0 8px 0',
            padding='0 0 8px 0',
            border_bottom='1px solid #e0e0e0'
        )
    )
    
    # Create category container
    container = widgets.VBox(
        [header] + package_widgets,
        layout=widgets.Layout(
            width='100%',
            margin='0 0 20px 0',
            padding='12px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            background='white'
        )
    )
    
    return {
        'widget': container,
        'checkboxes': checkboxes
    }

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str, message: str = None) -> None:
    """Update status widget for package with improved styling
    
    Args:
        ui_components: Dictionary containing UI components
        package_key: Key of the package to update
        status: Status to set (checking, installing, installed, error, warning, info, skipped)
        message: Optional message to show in tooltip
    """
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return
        
    checkboxes = ui_components['checkboxes']
    if package_key not in checkboxes:
        return
        
    checkbox = checkboxes[package_key]
    if not hasattr(checkbox, 'status'):
        return
        
    status_widget = checkbox.status
    
    # Define status styles: (icon, text_color, border_color, bg_color)
    status_styles = {
        'checking': ('⏳', '#FFA500', '#FFA500', '#FFF3E0'),  # Orange
        'installing': ('⏳', '#FFA500', '#FFA500', '#FFF3E0'),  # Orange
        'installed': ('✓', '#4CAF50', '#4CAF50', '#E8F5E9'),  # Green
        'error': ('✗', '#F44336', '#F44336', '#FFEBEE'),      # Red
        'warning': ('!', '#FFC107', '#FFC107', '#FFF8E1'),    # Yellow
        'info': ('ℹ️', '#2196F3', '#2196F3', '#E3F2FD'),      # Blue
        'skipped': ('⏭️', '#9E9E9E', '#9E9E9E', '#F5F5F5')    # Grey
    }
    
    # Get status style or default to grey
    icon, text_color, border_color, bg_color = status_styles.get(
        status.lower(), 
        ('', '#9E9E9E', '#e0e0e0', '#f9f9f9')
    )
    
    # Update status widget
    status_widget.value = f'<div style="color: {text_color}; font-weight: bold;">{icon}</div>'
    
    # Update tooltip if message is provided
    if message:
        status_widget.tooltip = message
    
    # Update checkbox style
    checkbox.layout.border = f'1px solid {border_color}'
    checkbox.layout.background_color = bg_color

def get_selected_packages(ui_components: Dict[str, Any]) -> list:
    """Get list of selected packages for installation"""
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return []
        
    checkboxes = ui_components['checkboxes']
    selected = []
    
    for pkg_key, checkbox in checkboxes.items():
        if checkbox.value:  # If checkbox is checked
            selected.append({
                'key': pkg_key,
                'name': checkbox.description,
                'optional': not checkbox.disabled
            })
    
    return selected

def reset_package_selections(ui_components: Dict[str, Any]) -> None:
    """Reset package selections to default values"""
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return
        
    checkboxes = ui_components['checkboxes']
    categories = get_package_categories()
    
    # Create a mapping of package keys to their default values
    defaults = {}
    for category in categories:
        for pkg in category['packages']:
            defaults[pkg['key']] = pkg.get('default', False)
    
    # Update checkboxes
    for pkg_key, checkbox in checkboxes.items():
        if pkg_key in defaults:
            checkbox.value = defaults[pkg_key]
            # Reset status
            if hasattr(checkbox, 'status'):
                checkbox.status.value = ''
                checkbox.status.tooltip = ''
            # Reset style
            checkbox.layout.border = '1px solid #e0e0e0'
            checkbox.layout.background_color = '#f9f9f9'
