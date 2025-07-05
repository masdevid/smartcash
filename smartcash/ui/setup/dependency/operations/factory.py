"""
Factory for creating operation handlers.
"""
from typing import Dict, Any, Type, Optional, TypeVar, List

from .base_operation import BaseOperationHandler
from .install_operation import InstallOperationHandler
from .update_operation import UpdateOperationHandler
from .uninstall_operation import UninstallOperationHandler
from .check_status_operation import CheckStatusOperationHandler


class OperationHandlerFactory:
    """Factory for creating operation handlers."""
    
    # Map of operation types to handler classes
    _HANDLERS = {
        'install': InstallOperationHandler,
        'update': UpdateOperationHandler,
        'uninstall': UninstallOperationHandler,
        'check_status': CheckStatusOperationHandler
    }
    
    @classmethod
    def create_handler(
        cls, 
        operation_type: str, 
        ui_components: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Optional[BaseOperationHandler]:
        """Create an operation handler for the specified operation type.
        
        Args:
            operation_type: Type of operation (install/update/uninstall/check_status)
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            
        Returns:
            Operation handler instance or None if type is invalid
        """
        handler_class = cls._HANDLERS.get(operation_type)
        if not handler_class:
            return None
            
        return handler_class(ui_components, config)
    
    @classmethod
    def get_operation_types(cls) -> List[str]:
        """Get list of supported operation types.
        
        Returns:
            List of supported operation type strings
        """
        return list(cls._HANDLERS.keys())
    
    @classmethod
    def is_valid_operation(cls, operation_type: str) -> bool:
        """Check if an operation type is valid.
        
        Args:
            operation_type: Operation type to check
            
        Returns:
            True if the operation type is valid, False otherwise
        """
        return operation_type in cls._HANDLERS
