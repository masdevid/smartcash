
"""
File: smartcash/ui/setup/dependency/__init__.py
Deskripsi: Package exports untuk dependency management module
"""

from typing import Dict, Any
from .dependency_initializer import DependencyInitializer

def initialize_dependency_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Initialize dependency management UI
    
    Args:
        config: Konfigurasi dependency (opsional)
        
    Returns:
        Dictionary berisi UI components
    """
    initializer = DependencyInitializer()
    return initializer.initialize_ui(config=config)

# Export untuk memudahkan import
__all__ = [
    'DependencyInitializer',
    'initialize_dependency_ui'
]