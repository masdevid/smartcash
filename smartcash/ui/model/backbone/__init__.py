"""
Backbone Module - Model backbone functionality and interfaces

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/backbone/__init__.py
"""

from .backbone_uimodule import BackboneUIModule
from .backbone_ui_factory import BackboneUIFactory, create_backbone_display

def initialize_backbone_ui(config=None, **kwargs):
    """
    Initialize and display the backbone UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        The created UI module or None if failed
    """
    return BackboneUIFactory.create_and_display_backbone(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'BackboneUIModule',
    'BackboneUIFactory',
    'initialize_backbone_ui',
    'create_backbone_display'
]
