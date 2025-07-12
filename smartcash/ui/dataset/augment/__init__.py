"""
File: smartcash/ui/dataset/augment/__init__.py
Description: Main exports for augment module using new UIModule pattern

This module provides the main interface for the augment module using the new
UIModule architecture while preserving all business logic and UI flow.
"""

# ==================== NEW UIMODULE API ====================

from .augment_uimodule import (
    AugmentUIModule,
    create_augment_uimodule,
    get_augment_uimodule,
    reset_augment_uimodule,
    initialize_augment_ui,
    get_augment_components
)

# ==================== CORE COMPONENTS ====================

from .components.augment_ui import create_augment_ui
from .configs.augment_config_handler import AugmentConfigHandler
from .operations.augment_operation_manager import AugmentOperationManager

# ==================== CONVENIENCE FUNCTIONS ====================

def display_augment_ui(
    config=None,
    **kwargs
) -> None:
    """
    Display augmentation UI with UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_augment_ui(
        config=config,
        display=True,
        **kwargs
    )

# ==================== EXPORTS ====================

__all__ = [
    # NEW UIMODULE API
    'AugmentUIModule',
    'create_augment_uimodule',
    'get_augment_uimodule',
    'reset_augment_uimodule',
    'initialize_augment_ui',
    'get_augment_components',
    'display_augment_ui',
    
    # CORE COMPONENTS
    'create_augment_ui',
    'AugmentConfigHandler',
    'AugmentOperationManager'
]