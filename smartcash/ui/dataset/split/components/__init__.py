"""
Dataset Split UI Components.

This module contains UI components for the dataset split functionality,
organized in a container-based pattern.
"""

from .split_ui import create_split_ui 
from .ratio_section import create_ratio_section
from .path_section import create_path_section
from .advanced_section import create_advanced_section

__all__ = [
    'create_split_ui',
    'create_ratio_section',
    'create_path_section',
    'create_advanced_section'
]
