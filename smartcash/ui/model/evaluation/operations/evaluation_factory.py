"""
File: smartcash/ui/model/evaluation/operations/evaluation_factory.py
Description: Factory functions for creating evaluation operations.
"""

from typing import Dict, Any, Callable, Optional, TYPE_CHECKING, Union

from .evaluation_all_operation import EvaluationAllOperation
from .evaluation_position_operation import EvaluationPositionOperation  
from .evaluation_lighting_operation import EvaluationLightingOperation

if TYPE_CHECKING:
    from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule


class EvaluationOperationFactory:
    """
    Factory class for creating evaluation operations.
    
    Provides centralized creation and management of evaluation operation instances
    with consistent configuration and callback handling.
    """
    
    # Operation type mapping
    OPERATION_TYPES = {
        'all_scenarios': EvaluationAllOperation,
        'position_only': EvaluationPositionOperation,
        'lighting_only': EvaluationLightingOperation
    }

    @classmethod
    def create_operation(
        cls,
        operation_type: str,
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Union[EvaluationAllOperation, EvaluationPositionOperation, EvaluationLightingOperation]:
        """
        Create an evaluation operation instance.
        
        Args:
            operation_type: Type of operation ('all_scenarios', 'position_only', 'lighting_only')
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            Operation instance of the specified type
            
        Raises:
            ValueError: If operation_type is not supported
        """
        if operation_type not in cls.OPERATION_TYPES:
            supported_types = ', '.join(cls.OPERATION_TYPES.keys())
            raise ValueError(f"Unsupported operation type: {operation_type}. Supported types: {supported_types}")
        
        operation_class = cls.OPERATION_TYPES[operation_type]
        return operation_class(ui_module, config, callbacks)

    @classmethod
    def get_supported_operations(cls) -> list:
        """
        Get list of supported operation types.
        
        Returns:
            List of supported operation type strings
        """
        return list(cls.OPERATION_TYPES.keys())

    @classmethod
    def create_all_scenarios_operation(
        cls,
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> EvaluationAllOperation:
        """
        Create an all scenarios evaluation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            EvaluationAllOperation instance
        """
        return cls.create_operation('all_scenarios', ui_module, config, callbacks)

    @classmethod
    def create_position_operation(
        cls,
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> EvaluationPositionOperation:
        """
        Create a position evaluation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            EvaluationPositionOperation instance
        """
        return cls.create_operation('position_only', ui_module, config, callbacks)

    @classmethod
    def create_lighting_operation(
        cls,
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> EvaluationLightingOperation:
        """
        Create a lighting evaluation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            EvaluationLightingOperation instance
        """
        return cls.create_operation('lighting_only', ui_module, config, callbacks)


# Convenience factory functions
def create_all_scenarios_operation(
    ui_module: 'EvaluationUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> EvaluationAllOperation:
    """
    Create an all scenarios evaluation operation.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary
        callbacks: Optional callbacks
        
    Returns:
        EvaluationAllOperation instance
    """
    return EvaluationOperationFactory.create_all_scenarios_operation(ui_module, config, callbacks)


def create_position_operation(
    ui_module: 'EvaluationUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> EvaluationPositionOperation:
    """
    Create a position evaluation operation.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary
        callbacks: Optional callbacks
        
    Returns:
        EvaluationPositionOperation instance
    """
    return EvaluationOperationFactory.create_position_operation(ui_module, config, callbacks)


def create_lighting_operation(
    ui_module: 'EvaluationUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> EvaluationLightingOperation:
    """
    Create a lighting evaluation operation.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary
        callbacks: Optional callbacks
        
    Returns:
        EvaluationLightingOperation instance
    """
    return EvaluationOperationFactory.create_lighting_operation(ui_module, config, callbacks)