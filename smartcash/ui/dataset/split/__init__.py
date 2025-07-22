"""
Split Module - Data handling and processing for split

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/split/__init__.py
"""

from .split_uimodule import SplitUIModule
from .split_ui_factory import SplitUIFactory, create_split_display

def initialize_split_ui(config=None, **kwargs):
    """
    Initialize and display the split UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    SplitUIFactory.create_and_display_split(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'SplitUIModule',
    'SplitUIFactory',
    'initialize_split_ui',
    'create_split_display'
]
