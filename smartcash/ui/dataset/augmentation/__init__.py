"""
Augmentation Module - Data handling and processing for augmentation

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/augmentation/__init__.py
"""

from .augmentation_uimodule import AugmentationUIModule
from .augmentation_ui_factory import AugmentationUIFactory, create_augmentation_display

def initialize_augmentation_ui(config=None, **kwargs):
    """
    Initialize and display the augmentation UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    AugmentationUIFactory.create_and_display_augmentation(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'AugmentationUIModule',
    'AugmentationUIFactory',
    'initialize_augmentation_ui',
    'create_augmentation_display'
]
