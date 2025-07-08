"""
File: smartcash/ui/setup/colab/operations/factory.py
Description: Factory for creating colab operation handlers
"""

from typing import Dict, Any, Optional
from smartcash.ui.components.operation_container import OperationContainer
from .operation_manager import ColabOperationManager


class OperationHandlerFactory:
    """Factory for creating colab operation handlers."""
    
    @classmethod
    def create_handler(cls, 
                      config: Dict[str, Any],
                      operation_container: Optional[OperationContainer] = None,
                      max_workers: int = 2) -> ColabOperationManager:
        """Create a colab operation manager.
        
        Args:
            config: Configuration dictionary
            operation_container: Optional OperationContainer for UI integration
            max_workers: Maximum number of worker threads
            
        Returns:
            ColabOperationManager instance
        """
        return ColabOperationManager(
            config=config,
            operation_container=operation_container,
            max_workers=max_workers
        )
    
    @classmethod
    def get_available_operations(cls) -> list:
        """Get list of available operation types.
        
        Returns:
            List of available operation type names
        """
        return [
            'init',
            'drive', 
            'symlink',
            'folders',
            'config',
            'env',
            'verify',
            'full_setup'
        ]
    
    @classmethod
    def is_operation_available(cls, operation_type: str) -> bool:
        """Check if an operation type is available.
        
        Args:
            operation_type: Operation type to check
            
        Returns:
            True if operation is available, False otherwise
        """
        return operation_type in cls.get_available_operations()