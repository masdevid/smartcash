"""
Dataset package for SmartCash UI.

This package contains modules for dataset manipulation including augmentation, preprocessing, and visualization.
"""

# Import submodules to make them available when importing the package
from smartcash.ui.dataset import augmentation
from smartcash.ui.dataset import visualization

__all__ = ['augmentation', 'visualization']
