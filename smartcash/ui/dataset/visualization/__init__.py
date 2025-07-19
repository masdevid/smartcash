"""
Visualization Module - Model visualization functionality and interfaces

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/visualization/__init__.py
"""

from .visualization_uimodule import VisualizationUIModule
from .visualization_ui_factory import VisualizationUIFactory, create_visualization_display

def initialize_visualization_ui(config=None, **kwargs):
    """
    Initialize and display the visualization UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    VisualizationUIFactory.create_and_display_visualization(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'VisualizationUIModule',
    'VisualizationUIFactory',
    'initialize_visualization_ui',
    'create_visualization_display'
]
