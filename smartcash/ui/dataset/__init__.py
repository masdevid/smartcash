"""
Dataset Module - Comprehensive dataset management for SmartCash

This module provides interfaces for dataset operations including augmentation, preprocessing,
splitting, and visualization.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/__init__.py
"""

# Import submodules to make them available through the package
from . import augmentation
from . import downloader
from . import preprocessing
from . import split
from . import visualization

# Export main classes and functions
__all__ = [
    # Submodules
    'augmentation',
    'downloader',
    'preprocessing',
    'split',
    'visualization'
]