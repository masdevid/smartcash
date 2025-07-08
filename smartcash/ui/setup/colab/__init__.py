"""
file_path: smartcash/ui/setup/colab/__init__.py
Deskripsi: Modul utama untuk inisialisasi dan manajemen UI di lingkungan Google Colab.
"""

from .colab_initializer import ColabEnvInitializer, initialize_colab_ui

__version__ = "1.0.0"
__all__ = ["ColabEnvInitializer", "initialize_colab_ui"]

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
