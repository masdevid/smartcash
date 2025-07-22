"""
Pretrained Module - Model pretrained functionality and interfaces

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/pretrained/__init__.py
"""

from .pretrained_uimodule import PretrainedUIModule
from .pretrained_ui_factory import PretrainedUIFactory, create_pretrained_display
from .operations.pretrained_factory import PretrainedOperationFactory, PretrainedOperationType

def initialize_pretrained_ui(config=None, **kwargs):
    """
    Initialize and display the pretrained UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    PretrainedUIFactory.create_and_display_pretrained(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'PretrainedUIModule',
    'PretrainedUIFactory',
    'PretrainedOperationFactory',
    'PretrainedOperationType',
    'initialize_pretrained_ui',
    'create_pretrained_display'
]
