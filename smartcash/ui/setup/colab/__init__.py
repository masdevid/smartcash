"""
Colab Setup Module - Configuration and initialization for colab environment

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/colab/__init__.py
"""

from .colab_uimodule import ColabUIModule
from .colab_ui_factory import ColabUIFactory, create_colab_display

def initialize_colab_ui(config=None, **kwargs):
    """
    Initialize and display the colab UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        The created UI module or None if failed
    """
    return ColabUIFactory.create_and_display_colab(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'ColabUIModule',
    'ColabUIFactory',
    'initialize_colab_ui',
    'create_colab_display'
]
