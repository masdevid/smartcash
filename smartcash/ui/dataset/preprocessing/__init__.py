"""
Preprocessing Module - Data handling and processing for preprocessing

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/preprocessing/__init__.py
"""

from .preprocessing_uimodule import PreprocessingUIModule
from .preprocessing_ui_factory import PreprocessingUIFactory, create_preprocessing_display

def initialize_preprocessing_ui(config=None, **kwargs):
    """
    Initialize and display the preprocessing UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    PreprocessingUIFactory.create_and_display_preprocessing(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'PreprocessingUIModule',
    'PreprocessingUIFactory',
    'initialize_preprocessing_ui',
    'create_preprocessing_display'
]
