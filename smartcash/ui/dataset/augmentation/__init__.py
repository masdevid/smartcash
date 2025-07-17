"""
File: smartcash/ui/dataset/augmentation/__init__.py
Description: Augmentation module exports using UIModule pattern.
"""

# Import simplified UIModule functions
from .augmentation_uimodule import (
    AugmentationUIModule,
    create_augmentation_uimodule,
    get_augmentation_uimodule,
    reset_augmentation_uimodule,
    initialize_augmentation_ui
)

__all__ = [
    # UIModule pattern
    "AugmentationUIModule",
    "create_augmentation_uimodule",
    "get_augmentation_uimodule",
    "reset_augmentation_uimodule",
    "initialize_augmentation_ui"
]
