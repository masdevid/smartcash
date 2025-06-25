# File: smartcash/ui/setup/dependency/utils/package_status_utils.py
# Deskripsi: Utilities untuk mengelola status package terpisah dari UI state

from typing import Dict, Any, List

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