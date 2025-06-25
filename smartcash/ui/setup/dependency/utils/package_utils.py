"""
Package Utility Functions.

This module provides utility functions for working with package configurations,
including filtering, searching, and validating package data.
"""
from typing import Any, Dict, List, Optional, Tuple
import os

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def filter_packages(
    packages: List[Dict[str, Any]],
    include_required: bool = True,
    include_optional: bool = True,
    include_installed: bool = True,
    include_not_installed: bool = True,
    search_term: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter a list of packages based on various criteria.
    
    Args:
        packages: List of package dictionaries to filter
        include_required: Include packages marked as required
        include_optional: Include packages not marked as required
        include_installed: Include packages marked as installed
        include_not_installed: Include packages not marked as installed
        search_term: Optional search term to filter package names and descriptions
        
    Returns:
        Filtered list of package dictionaries
    """
    if not packages:
        return []
    
    filtered = []
    
    for pkg in packages:
        # Skip if package doesn't match required/optional filter
        is_required = pkg.get('required', pkg.get('default', False))
        if (is_required and not include_required) or (not is_required and not include_optional):
            continue
            
        # Skip if package doesn't match installed filter
        is_installed = pkg.get('installed', False)
        if (is_installed and not include_installed) or (not is_installed and not include_not_installed):
            continue
            
        # Apply search term filter if provided
        if search_term:
            search_lower = search_term.lower()
            name_matches = search_lower in pkg.get('name', '').lower()
            desc_matches = search_lower in pkg.get('description', '').lower()
            
            if not (name_matches or desc_matches):
                continue
        
        filtered.append(pkg)
    
    return filtered

def get_package_details(
    package_key: str,
    categories: List[Dict[str, Any]],
    case_sensitive: bool = False
) -> Optional[Dict[str, Any]]:
    """Retrieve details for a specific package by its key or name.
    
    Args:
        package_key: The package key or name to search for
        categories: List of categories to search in
        case_sensitive: Whether the search should be case-sensitive
        
    Returns:
        The package details dictionary if found, None otherwise
    """
    if not package_key or not categories:
        return None
    
    # Normalize search key if case-insensitive
    search_key = package_key if case_sensitive else package_key.lower()
    
    for category in categories:
        for pkg in category.get('packages', []):
            # Check both key and name fields
            pkg_key = pkg.get('key', pkg.get('name', ''))
            pkg_name = pkg.get('name', '')
            
            # Compare based on case sensitivity
            if case_sensitive:
                key_matches = pkg_key == search_key
                name_matches = pkg_name == search_key
            else:
                key_matches = pkg_key.lower() == search_key
                name_matches = pkg_name.lower() == search_key
            
            if key_matches or name_matches:
                return pkg
    
    return None

def get_all_package_names(
    categories: List[Dict[str, Any]],
    include_keys: bool = True,
    include_names: bool = True,
    unique_only: bool = True
) -> List[str]:
    """Retrieve all package names and/or keys from the configuration.
    
    Args:
        categories: List of categories to search in
        include_keys: Whether to include package keys in the result
        include_names: Whether to include package names in the result
        unique_only: Whether to return only unique values
        
    Returns:
        List of package names and/or keys
    """       
    result = []
    
    for category in categories:
        for pkg in category.get('packages', []):
            if include_keys:
                if 'key' in pkg and pkg['key']:
                    result.append(pkg['key'])
                elif 'name' in pkg and pkg['name'] and not include_names:
                    # Use name as key if key is not available
                    result.append(pkg['name'])
            
            if include_names and 'name' in pkg and pkg['name']:
                result.append(pkg['name'])
    
    if unique_only:
        # Remove duplicates while preserving order
        seen = set()
        result = [x for x in result if not (x in seen or seen.add(x))]
    
    return result

def merge_configs(
    base_config: Dict[str, Any], 
    override_config: Dict[str, Any],
    list_merge_strategy: str = 'replace'
) -> Dict[str, Any]:
    """Merge two configuration dictionaries with support for nested structures.
    
    Args:
        base_config: The base configuration that will be overridden
        override_config: The configuration containing overrides
        list_merge_strategy: How to handle list merging. Can be one of:
            - 'replace': Replace the entire list (default)
            - 'extend': Extend the base list with override values
            - 'merge_unique': Merge unique values from both lists
            
    Returns:
        A new dictionary containing the merged configuration
    """
    if not isinstance(base_config, dict) or not isinstance(override_config, dict):
        return override_config if override_config is not None else base_config
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result:
            # Handle nested dictionaries recursively
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value, list_merge_strategy)
            # Handle lists based on merge strategy
            elif isinstance(result[key], list) and isinstance(value, list):
                if list_merge_strategy == 'extend':
                    result[key].extend(value)
                elif list_merge_strategy == 'merge_unique':
                    result[key] = list(dict.fromkeys(result[key] + value))
                else:  # 'replace' or unknown strategy
                    result[key] = value
            # Handle all other types by overriding
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a configuration dictionary against the expected schema.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, issues) where:
        - is_valid: Boolean indicating if the config is valid
        - issues: List of strings describing any validation issues found
    """
    issues = []
    
    # Check for required top-level fields
    required_fields = [
        'version', 
        'schema_version',
        'selected_packages',
        'categories',
        'package_manager',
        'python_path'
    ]
    
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
    
    # Validate version format if present
    if 'version' in config and not isinstance(config['version'], str):
        issues.append("Version must be a string")
    
    # Validate selected_packages is a list of strings
    selected_pkgs = config.get('selected_packages', [])
    if not isinstance(selected_pkgs, list) or \
       not all(isinstance(pkg, str) for pkg in selected_pkgs):
        issues.append("selected_packages must be a list of strings")
    
    # Validate categories structure
    categories = config.get('categories', [])
    if not isinstance(categories, list):
        issues.append("categories must be a list")
    else:
        for i, cat in enumerate(categories):
            if not isinstance(cat, dict):
                issues.append(f"Category at index {i} is not a dictionary")
                continue
                
            # Check required category fields
            for field in ['name', 'packages']:
                if field not in cat:
                    issues.append(f"Category at index {i} missing required field: {field}")
            
            # Validate packages in category
            if 'packages' in cat and isinstance(cat['packages'], list):
                for j, pkg in enumerate(cat['packages']):
                    if not isinstance(pkg, dict):
                        issues.append(f"Package at index {j} in category '{cat.get('name', 'unknown')}' is not a dictionary")
                        continue
                        
                    # Check required package fields
                    for pkg_field in ['key', 'name']:
                        if pkg_field not in pkg:
                            issues.append(
                                f"Package at index {j} in category '{cat.get('name', 'unknown')}' "
                                f"missing required field: {pkg_field}"
                            )
    
    # Check for any deprecated fields
    deprecated_fields = ['installation', 'analysis', 'ui_settings', 'advanced']
    for field in deprecated_fields:
        if field in config:
            issues.append(f"Deprecated field found: {field}")
    
    return len(issues) == 0, issues
