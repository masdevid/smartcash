"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/dependency/utils/package/status.py

Package status management utilities.

This module provides functions to manage and query package statuses in the UI.
"""

from typing import Dict, Any, List, Optional
from collections import Counter

# Absolute imports
from smartcash.ui.setup.dependency.utils.package.categories import get_package_categories

def analyze_packages(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze package status based on the provided configuration.
    
    Args:
        config: Configuration dictionary containing package information
        
    Returns:
        Dictionary mapping package names to their status information
    """
    from smartcash.ui.setup.dependency.utils.package.installer import get_installed_packages_dict
    
    # Get installed packages
    installed_packages = get_installed_packages_dict()
    
    # Initialize result dictionary
    result = {}
    
    # Get package dependencies from config
    dependencies = config.get('dependencies', {})
    
    # Check each package's installation status
    for pkg_name, pkg_info in dependencies.items():
        required = pkg_info.get('required', True)
        required_version = pkg_info.get('version', '')
        
        # Check if package is installed
        installed = pkg_name.lower() in (pkg.lower() for pkg in installed_packages.keys())
        
        # Get installed version if available
        installed_version = None
        for pkg, version in installed_packages.items():
            if pkg.lower() == pkg_name.lower():
                installed_version = version
                break
        
        # Add to results
        result[pkg_name] = {
            'installed': installed,
            'required': required,
            'required_version': required_version,
            'installed_version': installed_version
        }
    
    return result

__all__ = [
    'update_package_status',
    'update_package_status_by_name',
    'batch_update_package_status',
    'get_package_status_from_ui',
    'get_all_package_statuses',
    'reset_all_package_statuses',
    'filter_packages_by_status',
    'count_packages_by_status',
    'sync_package_status_with_system',
    'analyze_packages'
]

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str):
    """Update status package di UI selector dengan one-liner safe approach"""
    package_selector = ui_components.get('package_selector')
    package_selector and hasattr(package_selector, 'update_package_status') and package_selector.update_package_status(package_key, status)

def update_package_status_by_name(ui_components: Dict[str, Any], package_name: str, status: str):
    """Update package status berdasarkan nama dengan pencarian otomatis - one-liner"""
    [update_package_status(ui_components, package['key'], status)
     for category in __import__('smartcash.ui.setup.dependency.utils.package_categories', fromlist=['get_package_categories']).get_package_categories()
     for package in category['packages']
     if package['pip_name'].split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip().lower() == package_name.lower()]

def batch_update_package_status(ui_components: Dict[str, Any], status_mapping: Dict[str, str]):
    """Batch update package status dengan mapping - one-liner"""
    [update_package_status_by_name(ui_components, package_name, status) 
     for package_name, status in status_mapping.items()]

def get_package_status_from_ui(ui_components: Dict[str, Any], package_key: str) -> str:
    """Get current package status dari UI selector"""
    package_selector = ui_components.get('package_selector')
    if package_selector and hasattr(package_selector, 'get_package_status'):
        return package_selector.get_package_status(package_key)
    return "unknown"

def get_all_package_statuses(ui_components: Dict[str, Any]) -> Dict[str, str]:
    """Get semua package status dari UI selector - one-liner dict comprehension"""
    package_selector = ui_components.get('package_selector')
    if not package_selector or not hasattr(package_selector, 'get_all_statuses'):
        return {}
    return package_selector.get_all_statuses()

def reset_all_package_statuses(ui_components: Dict[str, Any]):
    """Reset semua package status ke default - one-liner"""
    package_selector = ui_components.get('package_selector')
    package_selector and hasattr(package_selector, 'reset_all_statuses') and package_selector.reset_all_statuses()

def filter_packages_by_status(ui_components: Dict[str, Any], target_status: str) -> List[str]:
    """Filter packages berdasarkan status tertentu - one-liner"""
    all_statuses = get_all_package_statuses(ui_components)
    return [pkg_key for pkg_key, status in all_statuses.items() if status == target_status]

def count_packages_by_status(ui_components: Dict[str, Any]) -> Dict[str, int]:
    """Count packages berdasarkan status - one-liner dengan Counter"""
    from collections import Counter
    all_statuses = get_all_package_statuses(ui_components)
    return dict(Counter(all_statuses.values()))

def sync_package_status_with_system(ui_components: Dict[str, Any], installed_packages: Dict[str, str]):
    """Sync package status dengan sistem yang terinstall"""
    from smartcash.ui.setup.dependency.utils.package_utils import extract_package_name_from_requirement
    
    # Get all packages dari UI
    try:
        get_package_categories = __import__('smartcash.ui.setup.dependency.utils.package_categories', 
                                          fromlist=['get_package_categories']).get_package_categories
        categories = get_package_categories()
    except ImportError:
        return
    
    # Update status berdasarkan sistem
    for category in categories:
        for package in category['packages']:
            pip_name = extract_package_name_from_requirement(package['pip_name'])
            status = "installed" if pip_name.lower() in [p.lower() for p in installed_packages.keys()] else "not_installed"
            update_package_status(ui_components, package['key'], status)