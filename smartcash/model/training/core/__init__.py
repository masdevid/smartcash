"""
Core training components for the unified training pipeline.

This package contains the refactored training components following
Single Responsibility Principle (SRP) design.
"""

from .phase_orchestrator import PhaseOrchestrator
from .training_executor import TrainingExecutor
from .validation_executor import ValidationExecutor
from .prediction_processor import PredictionProcessor
# mAP calculator modules removed - calculation disabled for performance
from .training_checkpoint_adapter import TrainingCheckpointAdapter
from .progress_manager import ProgressManager

__all__ = [
    'PhaseOrchestrator',
    'TrainingExecutor', 
    'ValidationExecutor',
    'PredictionProcessor',
    # mAP calculator classes removed
    'TrainingCheckpointAdapter',
    'ProgressManager'
]