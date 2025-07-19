"""
Setup Module - Environment and dependency setup for SmartCash

This module provides interfaces for setting up the SmartCash environment,
including dependency management and Colab environment configuration.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/__init__.py
"""
from importlib import import_module
from typing import Any

# List of available submodules
__all__ = ['colab', 'dependency']

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
