# File: smartcash/ui/pretrained/__init__.py
"""
File: smartcash/ui/pretrained/__init__.py
Deskripsi: Pretrained module untuk SmartCash dengan CommonInitializer pattern.

Modul ini menyediakan antarmuka untuk konfigurasi pretrained models YOLOv5s
yang telah disederhanakan untuk fokus pada currency detection.
"""

# Ekspor utama
from .pretrained_initializer import (
    PretrainedInitializer,
    initialize_pretrained_ui
)

# Aliases for backward compatibility
init_pretrained = initialize_pretrained_ui

__all__ = [
    # Classes
    'PretrainedInitializer',
    
    # Factory functions
    'initialize_pretrained_ui',
    
    # Aliases for backward compatibility
    'init_pretrained'
]