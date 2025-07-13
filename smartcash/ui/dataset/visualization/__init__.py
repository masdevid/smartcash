"""
File: smartcash/ui/dataset/visualization/__init__.py
Description: Visualization module for dataset analysis
"""

# Import new module API
from .visualization_module import (
    VisualizationUIModule,
    create_visualization_module,
    get_visualization_module,
    display_visualization_ui
)

# Import legacy API for backward compatibility
from .visualization_initializer import initialize_visualization_ui

# Export public API
__all__ = [
    # New API
    'VisualizationUIModule',
    'create_visualization_module',
    'get_visualization_module',
    'display_visualization_ui',
    
    # Legacy API (deprecated)
    'initialize_visualization_ui'
]
