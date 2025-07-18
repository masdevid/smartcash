"""
File: smartcash/ui/model/backbone/__init__.py
Description: Main exports for backbone module using new UIModule pattern

This module provides the main interface for the backbone model configuration module using the new
UIModule architecture while preserving backbone selection and early training pipeline functionality.
"""

# ==================== NEW UIMODULE API ====================

from .backbone_uimodule import (
    BackboneUIModule,
    create_backbone_uimodule,
    get_backbone_uimodule,
    reset_backbone_uimodule,
    initialize_backbone_ui,
    get_backbone_components
)

# ==================== CORE COMPONENTS ====================

from .components.backbone_ui import create_backbone_ui, update_model_summary
from .configs.backbone_config_handler import BackboneConfigHandler
from .operations.backbone_factory import BackboneOperationFactory

# ==================== CONVENIENCE FUNCTIONS ====================

def display_backbone_ui(
    config=None,
    **kwargs
) -> None:
    """
    Display backbone UI with UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_backbone_ui(
        config=config,
        display=True,
        **kwargs
    )

# ==================== EXPORTS ====================

__all__ = [
    # NEW UIMODULE API
    'BackboneUIModule',
    'create_backbone_uimodule',
    'get_backbone_uimodule',
    'reset_backbone_uimodule',
    'initialize_backbone_ui',
    'get_backbone_components',
    'display_backbone_ui',
    
    # CORE COMPONENTS
    'create_backbone_ui',
    'update_model_summary',
    'BackboneConfigHandler',
    'BackboneOperationFactory'
]