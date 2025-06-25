"""
Package Category Utilities.

This module provides access to package categories and their configurations.
It defines the default package categories and provides helper functions
for working with package categories.
"""

from typing import List, Dict, Any, Optional
import copy
from functools import lru_cache

# Default package categories
DEFAULT_CATEGORIES = [
    {
        'name': 'core',
        'display_name': 'Core Dependencies',
        'description': 'Essential packages required for basic functionality',
        'packages': [
            {'name': 'torch', 'required': True},
            {'name': 'torchvision', 'required': True},
            {'name': 'ultralytics', 'required': True},
        ]
    },
    {
        'name': 'data_science',
        'display_name': 'Data Science',
        'description': 'Libraries for data analysis and scientific computing',
        'packages': [
            {'name': 'numpy', 'required': False},
            {'name': 'pandas', 'required': False},
            {'name': 'scikit-learn', 'required': False},
        ]
    },
    {
        'name': 'visualization',
        'display_name': 'Visualization',
        'description': 'Tools for data visualization',
        'packages': [
            {'name': 'matplotlib', 'required': False},
            {'name': 'seaborn', 'required': False},
            {'name': 'plotly', 'required': False},
        ]
    },
    {
        'name': 'development',
        'display_name': 'Development Tools',
        'description': 'Tools for development and debugging',
        'packages': [
            {'name': 'pytest', 'required': False},
            {'name': 'black', 'required': False},
            {'name': 'mypy', 'required': False},
        ]
    }
]

__all__ = [
    'get_package_categories',
    'get_category_by_name',
    'get_package_info',
    'DEFAULT_CATEGORIES'
]

@lru_cache(maxsize=1)
def get_package_categories() -> List[Dict[str, Any]]:
    """Get a deep copy of all package categories with their configurations.
    
    The result is cached for better performance.
    
    Returns:
        List[Dict[str, Any]]: A list of package category configurations.
        Each category is a dictionary containing package information.
    """
    return copy.deepcopy(DEFAULT_CATEGORIES)


def get_category_by_name(category_name: str) -> Optional[Dict[str, Any]]:
    """Get a category by its name.
    
    Args:
        category_name: The name of the category to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: The category dictionary if found, None otherwise
    """
    for category in get_package_categories():
        if category.get('name') == category_name:
            return copy.deepcopy(category)
    return None


def get_package_info(package_name: str) -> Optional[Dict[str, Any]]:
    """Get package information by name across all categories.
    
    Args:
        package_name: The name of the package to find
        
    Returns:
        Optional[Dict[str, Any]]: The package info if found, None otherwise
    """
    for category in get_package_categories():
        for pkg in category.get('packages', []):
            if pkg.get('name') == package_name:
                return copy.deepcopy(pkg)
    return None
