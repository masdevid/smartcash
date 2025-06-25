"""
File: smartcash/ui/setup/dependency/utils/ui/utils.py

UI utility functions for dependency management.

This module provides utility functions for managing UI components and interactions.
"""

# Standard library imports
from typing import Any, Dict, List, Optional

# Absolute imports
from smartcash.ui.setup.dependency.utils.package.categories import get_package_categories

__all__ = ['get_selected_packages', 'reset_package_selections']

def get_selected_packages(ui_components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get list of selected packages for installation
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        List of selected package dictionaries with keys: key, name, optional
    """
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
    """Reset package selections to default values
    
    Args:
        ui_components: Dictionary containing UI components
    """
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return
        
    from smartcash.ui.setup.dependency.utils.package_categories import get_package_categories
    
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
