"""
Factory for creating pretrained model operations.
"""

from typing import Dict, Any, Type
from enum import Enum

from .pretrained_base_operation import PretrainedBaseOperation
from .pretrained_download_operation import PretrainedDownloadOperation
from .pretrained_validate_operation import PretrainedValidateOperation
from .pretrained_refresh_operation import PretrainedRefreshOperation
from .pretrained_cleanup_operation import PretrainedCleanupOperation


class PretrainedOperationType(Enum):
    """Available pretrained operation types."""
    DOWNLOAD = "download"
    VALIDATE = "validate"
    REFRESH = "refresh"
    CLEANUP = "cleanup"


class PretrainedOperationFactory:
    """Factory class for creating pretrained model operations."""
    
    # Registry of operation types to classes
    _operation_registry: Dict[str, Type[PretrainedBaseOperation]] = {
        PretrainedOperationType.DOWNLOAD.value: PretrainedDownloadOperation,
        PretrainedOperationType.VALIDATE.value: PretrainedValidateOperation,
        PretrainedOperationType.REFRESH.value: PretrainedRefreshOperation,
        PretrainedOperationType.CLEANUP.value: PretrainedCleanupOperation,
    }
    
    @classmethod
    def create_operation(
        cls, 
        operation_type: str, 
        ui_components: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> PretrainedBaseOperation:
        """Create an operation instance.
        
        Args:
            operation_type: Type of operation to create
            ui_components: UI components dictionary
            config: Configuration dictionary
            
        Returns:
            PretrainedBaseOperation instance
            
        Raises:
            ValueError: If operation type is not supported
        """
        operation_class = cls._operation_registry.get(operation_type)
        
        if not operation_class:
            available_ops = list(cls._operation_registry.keys())
            raise ValueError(f"Unknown operation type: {operation_type}. Available: {available_ops}")
        
        return operation_class(ui_components, config)
    
    @classmethod
    def get_available_operations(cls) -> list:
        """Get list of available operation types.
        
        Returns:
            List of available operation type strings
        """
        return list(cls._operation_registry.keys())
    
    @classmethod
    def register_operation(cls, operation_type: str, operation_class: Type[PretrainedBaseOperation]):
        """Register a new operation type.
        
        Args:
            operation_type: String identifier for the operation
            operation_class: Class that implements the operation
        """
        cls._operation_registry[operation_type] = operation_class
    
    @classmethod
    def execute_operation(
        cls, 
        operation_type: str, 
        ui_components: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create and execute an operation in one call.
        
        Args:
            operation_type: Type of operation to execute
            ui_components: UI components dictionary
            config: Configuration dictionary
            
        Returns:
            Operation result dictionary
        """
        try:
            operation = cls.create_operation(operation_type, ui_components, config)
            return operation.execute_operation()
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to execute {operation_type} operation: {e}'
            }