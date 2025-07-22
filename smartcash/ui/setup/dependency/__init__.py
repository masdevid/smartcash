"""
Dependency Setup Module - Configuration and initialization for dependency environment

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/dependency/__init__.py
"""

from .dependency_uimodule import DependencyUIModule
from .dependency_ui_factory import DependencyUIFactory, create_dependency_display

def initialize_dependency_ui(config=None, **kwargs):
    """
    Initialize and display the dependency UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    DependencyUIFactory.create_and_display_dependency(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'DependencyUIModule',
    'DependencyUIFactory',
    'initialize_dependency_ui',
    'create_dependency_display'
]
