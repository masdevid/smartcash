"""
File: smartcash/ui/dataset/split/__init__.py
Description: Main exports for split module using new UIModule pattern

This module provides the main interface for the split configuration module using the new
UIModule architecture while preserving all configuration functionality.
"""

# ==================== NEW UIMODULE API ====================

from .split_uimodule import (
    SplitUIModule,
    create_split_uimodule,
    get_split_uimodule,
    reset_split_uimodule,
    initialize_split_ui,
    get_split_components
)

"""SmartCash Dataset Split UI Module.

This module provides UI components for dataset splitting functionality.
"""

# ==================== CORE COMPONENTS ====================

from .components.split_ui import create_split_ui
from .components.ratio_section import create_ratio_section
from .components.path_section import create_path_section
from .components.advanced_section import create_advanced_section
from .configs.split_config_handler import SplitConfigHandler

__all__ = [
    'create_split_ui',
    'create_ratio_section',
    'create_path_section',
    'create_advanced_section',
    'SplitConfigHandler'
]

# ==================== CONVENIENCE FUNCTIONS ====================

def display_split_ui(
    config=None,
    **kwargs
) -> None:
    """
    Display split UI with UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_split_ui(
        config=config,
        display=True,
        **kwargs
    )

# ==================== EXPORTS ====================

__all__ = [
    # NEW UIMODULE API
    'SplitUIModule',
    'create_split_uimodule',
    'get_split_uimodule',
    'reset_split_uimodule',
    'initialize_split_ui',
    'get_split_components',
    'display_split_ui',
    
    # CORE COMPONENTS
    'create_split_ui',
    'SplitConfigHandler'
]