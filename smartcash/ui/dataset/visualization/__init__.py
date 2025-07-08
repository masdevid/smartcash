"""
File: smartcash/ui/dataset/visualization/__init__.py
Description: Visualization module for dataset analysis
"""

from .visualization_initializer import VisualizationInitializer, initialize_visualization

# Only expose the initializer
__all__ = ['VisualizationInitializer', 'initialize_visualization']
