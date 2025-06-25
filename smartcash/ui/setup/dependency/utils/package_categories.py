"""
File: smartcash/ui/setup/dependency/utils/package_categories.py
Deskripsi: Package category utilities to avoid circular imports
"""

from typing import List, Dict, Any
from smartcash.ui.setup.dependency.handlers.defaults import PACKAGE_CATEGORIES

def get_package_categories() -> List[Dict[str, Any]]:
    """Get package categories with their configurations
    
    Returns:
        List of package category configurations
    """
    # Return a deep copy to prevent accidental modifications
    import copy
    return copy.deepcopy(PACKAGE_CATEGORIES)
