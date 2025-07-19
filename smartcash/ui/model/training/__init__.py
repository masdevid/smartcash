"""
Training Module - Model training functionality and interfaces

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/training/__init__.py
"""

from .training_uimodule import TrainingUIModule
from .training_ui_factory import TrainingUIFactory, create_training_display

def initialize_training_ui(config=None, **kwargs):
    """
    Initialize and display the training UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    TrainingUIFactory.create_and_display_training(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'TrainingUIModule',
    'TrainingUIFactory',
    'initialize_training_ui',
    'create_training_display'
]
