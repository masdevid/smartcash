"""
file_path: smartcash/ui/setup/colab/__init__.py
Deskripsi: Modul `colab` menyediakan antarmuka utama untuk inisialisasi dan manajemen
komponen UI konfigurasi lingkungan khusus Google Colab di SmartCash.

Modul ini merupakan hasil refaktor dari `smartcash.ui.setup.env_config` agar penamaan
lebih sesuai dengan konteks penggunaannya.

Ekspor:
    - constants: Modul berisi konstanta yang dibutuhkan untuk proses setup di Colab.
    - handlers: Modul berisi handler untuk manajemen konfigurasi dan setup
    - components: Modul berisi komponen UI untuk konfigurasi Colab
    - utils: Modul berisi utilitas untuk konfigurasi Colab
    - colab_initializer: Modul untuk inisialisasi konfigurasi Colab
"""

__version__ = "1.0.0"

from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Optional, List

# Public API -----------------------------------------------------------

# Lazy loading implementation
_import_structure = {
    "constants": ["*"],
    "handlers": ["*"],
    "components": ["*"],
    "utils": ["*"],
    "colab_initializer": ["initialize_colab_ui"],
}

# Cache for lazy-loaded modules
_module_cache: Dict[str, ModuleType] = {}

def _lazy_import(name: str) -> ModuleType:
    """Lazy import a module with caching.
    
    Args:
        name: Name of the module to import (without the package prefix)
        
    Returns:
        The imported module
    """
    if name not in _module_cache:
        full_name = f"smartcash.ui.setup.colab.{name}"
        _module_cache[name] = import_module(full_name)
    return _module_cache[name]

def __getattr__(name: str) -> Any:
    """Lazy load modules on attribute access.
    
    This allows us to use dot notation to access submodules without
    importing them until they're actually needed.
    
    Args:
        name: Name of the module or attribute to get
        
    Returns:
        The requested module or raises AttributeError if not found
    """
    if name in _import_structure:
        module = _lazy_import(name)
        # Update globals to avoid repeated lookups
        globals()[name] = module
        return module
    
    # Check if it's an attribute of a lazy-loaded module
    for module_name, attrs in _import_structure.items():
        if name in attrs or "*" in attrs:
            module = _lazy_import(module_name)
            if hasattr(module, name):
                attr = getattr(module, name)
                globals()[name] = attr
                return attr
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> List[str]:
    """Return list of available attributes."""
    # Get all explicitly exported names
    all_names = list(_import_structure.keys())
    
    # Add all attributes from __all__ of submodules with wildcard imports
    for module_name, attrs in _import_structure.items():
        if "*" in attrs:
            try:
                module = _lazy_import(module_name)
                if hasattr(module, "__all__"):
                    all_names.extend(module.__all__)
            except ImportError:
                pass
    
    return sorted(set(all_names + __all__))

# Export all public API
__all__: list[str] = [
    "constants",
    "handlers",
    "initialize_colab_ui",
]
