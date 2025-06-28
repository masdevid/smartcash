"""
File: smartcash/ui/model/backbone/__init__.py
Deskripsi: Backbone model UI module exports
"""

from .backbone_init import BackboneInitializer
from .components import (
    create_backbone_child_components,
    create_model_form,
    create_config_summary,
    update_config_summary
)

def initialize_backbone_ui(config: dict = None):
    """Initialize and display the backbone model UI.
    
    Args:
        config: Optional configuration dictionary
    """
    backbone_initializer = BackboneInitializer(config=config)
    return backbone_initializer.initialize()

__all__ = [
    # Main initializer
    'BackboneInitializer',
    'initialize_backbone_ui',
    
    # Components
    'create_backbone_child_components',
    'create_model_form',
    'create_config_summary',
    'update_config_summary',
]