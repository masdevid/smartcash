"""
File: smartcash/ui/model/pretrained/__init__.py
Description: Main exports for pretrained module using new UIModule pattern

This module provides the main interface for the pretrained models module using the new
UIModule architecture while preserving all model management functionality.
"""

# ==================== NEW UIMODULE API ====================

from .pretrained_uimodule import (
    PretrainedUIModule,
    create_pretrained_uimodule,
    get_pretrained_uimodule,
    reset_pretrained_uimodule,
    initialize_pretrained_ui,
    get_pretrained_components
)

# ==================== CORE COMPONENTS ====================

from .components.pretrained_ui import create_pretrained_ui
from .configs.pretrained_config_handler import PretrainedConfigHandler
from .operations.pretrained_operation_manager import PretrainedOperationManager

# ==================== CONVENIENCE FUNCTIONS ====================

def display_pretrained_ui(
    config=None,
    **kwargs
) -> None:
    """
    Display pretrained UI with UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_pretrained_ui(
        config=config,
        display=True,
        **kwargs
    )

# ==================== EXPORTS ====================

__all__ = [
    # NEW UIMODULE API
    'PretrainedUIModule',
    'create_pretrained_uimodule',
    'get_pretrained_uimodule',
    'reset_pretrained_uimodule',
    'initialize_pretrained_ui',
    'get_pretrained_components',
    'display_pretrained_ui',
    
    # CORE COMPONENTS
    'create_pretrained_ui',
    'PretrainedConfigHandler',
    'PretrainedOperationManager'
]