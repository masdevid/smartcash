"""
Core training components for the unified training pipeline.

This package contains the refactored training components following
Single Responsibility Principle (SRP) design.
"""

from .training_executor import TrainingExecutor
from .validation_executor import ValidationExecutor
from .prediction_processor import PredictionProcessor
# mAP calculator modules removed - calculation disabled for performance
# TrainingCheckpointAdapter removed - using CheckpointManager directly
from .progress_manager import ProgressManager

__all__ = [
    'TrainingExecutor', 
    'ValidationExecutor',
    'PredictionProcessor',
    # mAP calculator classes removed
    # TrainingCheckpointAdapter removed - using CheckpointManager directly
    'ProgressManager'
]