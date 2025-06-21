"""
File: smartcash/ui/dataset/augmentation/__init__.py
Deskripsi: Augmentation module exports dengan CommonInitializer pattern
"""

from smartcash.ui.dataset.augmentation.augmentation_initializer import (
    AugmentationInitializer,
    initialize_augmentation_ui
)

# Backward compatibility exports
from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler

__all__ = [
    'AugmentationInitializer',
    'initialize_augmentation_ui',
    'create_augmentation_main_ui', 
    'AugmentationConfigHandler'
]

# One-liner factory untuk ease of use
init_augmentation = initialize_augmentation_ui