"""
Dataset Module - Comprehensive dataset management for SmartCash

This module provides interfaces for dataset operations including augmentation, preprocessing,
splitting, and visualization. Submodules are lazily loaded on first access.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/__init__.py
"""
from importlib import import_module
from typing import Any

# List of available submodules
__all__ = [
    'augmentation',
    'downloader',
    'preprocessing',
    'split',
    'visualization'
]

# Module-level cache for imported submodules
_module_cache = {}

def __getattr__(name: str) -> Any:
    """Lazy load submodules on attribute access.
    
    Args:
        name: Name of the submodule to import
        
    Returns:
        The imported submodule
        
    Raises:
        AttributeError: If the requested submodule is not found
    """
    if name in __all__:
        if name not in _module_cache:
            _module_cache[name] = import_module(f'.{name}', __name__)
        return _module_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """List available attributes including lazy-loaded submodules."""
    return sorted(__all__ + list(_module_cache.keys()))