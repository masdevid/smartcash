"""
File: smartcash/ui/model/training/__init__.py
Description: Main exports for training module

This module provides the main interface for the training functionality.
Use TrainingUIFactory to create and manage TrainingUIModule instances.
"""

from .training_uimodule import TrainingUIModule
from .training_ui_factory import TrainingUIFactory, create_training_display

__all__ = [
    'TrainingUIModule',
    'TrainingUIFactory',
    'create_training_display'
]