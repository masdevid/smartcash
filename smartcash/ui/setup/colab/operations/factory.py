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
                      operation_type: str,
                      ui_components: Dict[str, Any],
                      config: Dict[str, Any],
                      max_workers: int = 2) -> ColabOperationManager:
        """Create a colab operation manager.
        
        Args:
            operation_type: Type of operation to create
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            max_workers: Maximum number of worker threads
            
        Returns:
            ColabOperationManager instance
        """
        # Get operation container from UI components if available
        operation_container = ui_components.get('operation_container')
        
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