"""
File: smartcash/ui/dataset/augmentation/operations/__init__.py
Description: Augmentation module operations.
"""

from .augmentation_base_operation import AugmentationBaseOperation, OperationPhase
from .augment_operation import AugmentOperation
from .augment_preview_operation import AugmentPreviewOperation
from .augment_status_operation import AugmentStatusOperation
from .augment_cleanup_operation import AugmentCleanupOperation
from .augment_factory import (
    create_operation,
    create_augment_operation,
    create_augment_preview_operation,
    create_augment_status_operation,
    create_augment_cleanup_operation
)

__all__ = [
    'AugmentationBaseOperation',
    'OperationPhase',
    'AugmentOperation',
    'AugmentPreviewOperation',
    'AugmentStatusOperation',
    'AugmentCleanupOperation',
    'create_operation',
    'create_augment_operation',
    'create_augment_preview_operation',
    'create_augment_status_operation',
    'create_augment_cleanup_operation'
]
