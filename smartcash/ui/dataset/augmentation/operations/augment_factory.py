"""
File: smartcash/ui/dataset/augmentation/operations/augment_factory.py
Description: Factory functions for creating augmentation operation instances.
"""

from typing import Dict, Any, Callable, Optional, Type, TypeVar, Generic
from .augmentation_base_operation import AugmentationBaseOperation
from .augment_operation import AugmentOperation
from .augment_preview_operation import AugmentPreviewOperation
from .augment_status_operation import AugmentStatusOperation
from .augment_cleanup_operation import AugmentCleanupOperation

# Type variable for operation classes
TOperation = TypeVar('TOperation', bound=AugmentationBaseOperation)

def create_operation(
    operation_class: Type[TOperation],
    ui_module: Any,
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> TOperation:
    """
    Generic factory function for creating operation instances.
    
    Args:
        operation_class: The operation class to instantiate
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the operation
        callbacks: Optional callbacks for operation events
        
    Returns:
        Instance of the specified operation class
    """
    return operation_class(ui_module, config, callbacks or {})

def create_augment_operation(
    ui_module: Any,
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> AugmentOperation:
    """Create an augmentation operation instance."""
    return create_operation(AugmentOperation, ui_module, config, callbacks)

def create_augment_preview_operation(
    ui_module: Any,
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> AugmentPreviewOperation:
    """Create a preview operation instance."""
    return create_operation(AugmentPreviewOperation, ui_module, config, callbacks)

def create_augment_status_operation(
    ui_module: Any,
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> AugmentStatusOperation:
    """Create a status operation instance."""
    return create_operation(AugmentStatusOperation, ui_module, config, callbacks)

def create_augment_cleanup_operation(
    ui_module: Any,
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> AugmentCleanupOperation:
    """Create a cleanup operation instance."""
    return create_operation(AugmentCleanupOperation, ui_module, config, callbacks)

def create_augmentation_operation(
    operation_type: str,
    ui_module: Any,
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> Optional[AugmentationBaseOperation]:
    """
    Create an augmentation operation instance by type.
    
    Args:
        operation_type: Type of operation ('augment', 'preview', 'status', 'cleanup')
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the operation
        callbacks: Optional callbacks for operation events
        
    Returns:
        Instance of the requested operation type or None if type not recognized
    """
    operation_map = {
        'augment': create_augment_operation,
        'preview': create_augment_preview_operation,
        'status': create_augment_status_operation,
        'cleanup': create_augment_cleanup_operation
    }
    
    factory_func = operation_map.get(operation_type.lower())
    if factory_func:
        return factory_func(ui_module, config, callbacks)
    return None

# Export all factory functions
__all__ = [
    'create_augment_operation',
    'create_augment_preview_operation',
    'create_augment_status_operation',
    'create_augment_cleanup_operation',
    'create_augmentation_operation'
]
