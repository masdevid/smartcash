"""
File: smartcash/ui/model/training/operations/__init__.py
Training operations package - Updated for unified training pipeline.
"""

# Unified training operation (primary)
from .unified_training_operation import UnifiedTrainingOperation

# Legacy operations (kept for compatibility)
from .training_factory import TrainingOperationFactory
from .training_base_operation import BaseTrainingOperation
from .training_start_operation import TrainingStartOperationHandler
from .training_stop_operation import TrainingStopOperationHandler
from .training_resume_operation import TrainingResumeOperationHandler
from .training_validate_operation import TrainingValidateOperationHandler

__all__ = [
    # Unified training operation (primary)
    'UnifiedTrainingOperation',
    
    # Legacy operations (compatibility)
    'TrainingOperationFactory',
    'BaseTrainingOperation',
    'TrainingStartOperationHandler',
    'TrainingStopOperationHandler',
    'TrainingResumeOperationHandler',
    'TrainingValidateOperationHandler'
]