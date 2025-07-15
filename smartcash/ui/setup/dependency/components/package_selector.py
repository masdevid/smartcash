"""
Package Selector Component

This module provides functionality to extract selected packages from the dependency UI components.
"""

from typing import Dict, Any, List
import re


def get_selected_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Get list of selected packages from package checkboxes.
    
    Args:
        ui_components: Dictionary of UI components containing package_checkboxes
        
    Returns:
        List of selected package names
    """
    selected_packages = []
    
    try:
        # Get package checkboxes from UI components
        package_checkboxes = ui_components.get('package_checkboxes', {})
        
        if not package_checkboxes:
            return selected_packages
        
        # Iterate through each category
        for category_key, checkboxes in package_checkboxes.items():
            if not checkboxes:
                continue
                
            # Extract selected packages from checkboxes
            for checkbox in checkboxes:
                if hasattr(checkbox, 'value') and checkbox.value:
                    # Extract package name from checkbox description
                    # Format is typically "package_name (version)"
                    if hasattr(checkbox, 'description'):
                        package_name = _extract_package_name(checkbox.description)
                        if package_name:
                            selected_packages.append(package_name)
        
        return list(set(selected_packages))  # Remove duplicates
        
    except Exception as e:
        # Log error but don't raise to avoid breaking operation
        from smartcash.ui.logger import get_module_logger
        logger = get_module_logger("smartcash.ui.setup.dependency.components.package_selector")
        logger.warning(f"Error getting selected packages: {e}")
        return []


def get_custom_packages_text(ui_components: Dict[str, Any]) -> str:
    """Get custom packages text from the custom packages textarea.
    
    Args:
        ui_components: Dictionary of UI components containing custom_packages widget
        
    Returns:
        String containing custom packages text
    """
    try:
        # Get custom packages widget from UI components
        custom_packages_widget = ui_components.get('custom_packages')
        
        if not custom_packages_widget:
            return ""
        
        # Extract text value from the widget
        if hasattr(custom_packages_widget, 'value'):
            return custom_packages_widget.value.strip()
        
        return ""
        
    except Exception as e:
        # Log error but don't raise to avoid breaking operation
        from smartcash.ui.logger import get_module_logger
        logger = get_module_logger("smartcash.ui.setup.dependency.components.package_selector")
        logger.warning(f"Error getting custom packages text: {e}")
        return ""


def get_all_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Get all packages (selected + custom) from UI components.
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        List of all package names
    """
    all_packages = []
    
    # Get selected packages from checkboxes
    selected_packages = get_selected_packages(ui_components)
    all_packages.extend(selected_packages)
    
    # Get custom packages from text area
    custom_packages_text = get_custom_packages_text(ui_components)
    if custom_packages_text:
        custom_packages = _parse_custom_packages(custom_packages_text)
        all_packages.extend(custom_packages)
    
    return list(set(all_packages))  # Remove duplicates


def _extract_package_name(description: str) -> str:
    """Extract package name from checkbox description.
    
    Args:
        description: Checkbox description in format "package_name (version)"
        
    Returns:
        Package name without version
    """
    if not description:
        return ""
    
    # Remove version info in parentheses
    # Example: "numpy (1.21.0)" -> "numpy"
    match = re.match(r'^([^(]+)', description.strip())
    if match:
        return match.group(1).strip()
    
    return description.strip()


def _parse_custom_packages(custom_text: str) -> List[str]:
    """Parse custom packages from text area.
    
    Args:
        custom_text: Multi-line text with package specifications
        
    Returns:
        List of package names/specifications
    """
    packages = []
    
    if not custom_text:
        return packages
    
    # Split by lines and process each
    for line in custom_text.split('\n'):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Add the package specification as-is (can include version requirements)
        packages.append(line)
    
    return packages


def get_packages_by_category(ui_components: Dict[str, Any], category: str) -> List[str]:
    """Get selected packages from a specific category.
    
    Args:
        ui_components: Dictionary of UI components
        category: Category name to get packages from
        
    Returns:
        List of selected package names from the specified category
    """
    selected_packages = []
    
    try:
        # Get package checkboxes from UI components
        package_checkboxes = ui_components.get('package_checkboxes', {})
        
        if not package_checkboxes or category not in package_checkboxes:
            return selected_packages
        
        # Get checkboxes for the specific category
        checkboxes = package_checkboxes[category]
        
        # Extract selected packages from checkboxes
        for checkbox in checkboxes:
            if hasattr(checkbox, 'value') and checkbox.value:
                # Extract package name from checkbox description
                if hasattr(checkbox, 'description'):
                    package_name = _extract_package_name(checkbox.description)
                    if package_name:
                        selected_packages.append(package_name)
        
        return selected_packages
        
    except Exception as e:
        # Log error but don't raise to avoid breaking operation
        from smartcash.ui.logger import get_module_logger
        logger = get_module_logger("smartcash.ui.setup.dependency.components.package_selector")
        logger.warning(f"Error getting packages for category {category}: {e}")
        return []


def parse_package_name(package_spec: str) -> str:
    """Parse package name from package specification.
    
    Args:
        package_spec: Package specification (e.g., "numpy>=1.20.0")
        
    Returns:
        Package name without version specifiers
    """
    if not package_spec:
        return ""
    
    # Remove version specifiers
    # Examples: "numpy>=1.20.0" -> "numpy", "pandas==1.3.0" -> "pandas"
    package_name = re.split(r'[><=!~]', package_spec.strip())[0].strip()
    
    # Clean up any additional characters
    package_name = re.sub(r'[^a-zA-Z0-9_.-]', '', package_name)
    
    return package_name


def validate_package_name(package_name: str) -> bool:
    """Validate if a package name is valid.
    
    Args:
        package_name: Package name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not package_name:
        return False
    
    # Check for basic validity
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_.-]*$', package_name):
        return False
    
    # Check for minimum length
    if len(package_name) < 1:
        return False
    
    # Check for maximum length
    if len(package_name) > 214:  # PyPI limit
        return False
    
    # Check for invalid patterns
    if package_name.startswith('.') or package_name.endswith('.'):
        return False
    
    if package_name.startswith('-') or package_name.endswith('-'):
        return False
    
    return True


def get_package_selection_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of package selection state.
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Dictionary with selection summary
    """
    try:
        selected_packages = get_selected_packages(ui_components)
        custom_packages_text = get_custom_packages_text(ui_components)
        custom_packages = _parse_custom_packages(custom_packages_text)
        
        # Get packages by category
        package_checkboxes = ui_components.get('package_checkboxes', {})
        categories = {}
        
        for category_key in package_checkboxes.keys():
            categories[category_key] = get_packages_by_category(ui_components, category_key)
        
        return {
            'selected_packages': selected_packages,
            'custom_packages': custom_packages,
            'total_selected': len(selected_packages),
            'total_custom': len(custom_packages),
            'total_packages': len(set(selected_packages + custom_packages)),
            'categories': categories,
            'custom_packages_text': custom_packages_text.strip()
        }
        
    except Exception as e:
        from smartcash.ui.logger import get_module_logger
        logger = get_module_logger("smartcash.ui.setup.dependency.components.package_selector")
        logger.error(f"Error getting package selection summary: {e}")
        return {
            'selected_packages': [],
            'custom_packages': [],
            'total_selected': 0,
            'total_custom': 0,
            'total_packages': 0,
            'categories': {},
            'custom_packages_text': '',
            'error': str(e)
        }