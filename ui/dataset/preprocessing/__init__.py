"""
Preprocessing module untuk SmartCash dengan CommonInitializer pattern.

Modul ini menyediakan antarmuka untuk preprocessing dataset dengan dukungan
konfigurasi yang fleksibel dan manajemen state yang kuat.
"""

# Ekspor utama
from .preprocessing_initializer import (
    PreprocessingInitializer,
    initialize_preprocessing_ui
)

# Aliases for backward compatibility
init_preprocessing = initialize_preprocessing_ui

__all__ = [
    # Classes
    'PreprocessingInitializer',
    
    # Factory functions
    'initialize_preprocessing_ui',
    
    # Aliases for backward compatibility
    'init_preprocessing'
]