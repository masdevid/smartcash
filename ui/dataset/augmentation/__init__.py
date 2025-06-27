"""
Augmentation module untuk SmartCash dengan CommonInitializer pattern.

Modul ini menyediakan antarmuka untuk augmentasi dataset dengan dukungan
konfigurasi yang fleksibel dan manajemen state yang kuat.
"""

# Ekspor utama
from smartcash.ui.dataset.augmentation.augmentation_initializer import (
    AugmentationInitializer,
    initialize_augmentation_ui
)
# Aliases for backward compatibility
init_augmentation = initialize_augmentation_ui

__all__ = [
    # Classes
    'AugmentationInitializer',
    
    # Factory functions
    'initialize_augmentation_ui',
    
    # Aliases for backward compatibility
    'init_augmentation'
]
