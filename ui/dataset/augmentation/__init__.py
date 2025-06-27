"""
Augmentation module untuk SmartCash dengan CommonInitializer pattern.

Modul ini menyediakan antarmuka untuk augmentasi dataset dengan dukungan
konfigurasi yang fleksibel dan manajemen state yang kuat.
"""

from typing import Dict, Any, Optional

# Ekspor utama
from smartcash.ui.dataset.augmentation.augmentation_initializer import (
    AugmentationInitializer,
    initialize_augmentation_ui
)

# Ekspor komponen untuk keperluan kustomisasi
from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler

# Backward compatibility
try:
    from smartcash.ui.dataset.augmentation.handlers.augmentation_handlers import setup_augmentation_handlers
except ImportError:
    # Handle case where augmentation_handlers is not available
    setup_augmentation_handlers = None

__all__ = [
    # Classes
    'AugmentationInitializer',
    'AugmentationConfigHandler',
    
    # Factory functions
    'initialize_augmentation_ui',
    'create_augmentation_main_ui',
    'setup_augmentation_handlers',
    
    # Aliases for backward compatibility
    'init_augmentation'
]

# Aliases for backward compatibility
init_augmentation = initialize_augmentation_ui

def get_version() -> str:
    """Dapatkan versi modul augmentasi.
    
    Returns:
        String versi dalam format 'x.y.z'
    """
    return "1.0.0"  # TODO: Ambil dari __version__ atau setup.py