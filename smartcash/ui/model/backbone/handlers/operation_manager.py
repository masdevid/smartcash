"""
File: smartcash/ui/model/backbone/handlers/operation_manager.py
Description: Operation manager for coordinating backbone operations
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.logger import get_module_logger
from ..operations import ValidateOperation, LoadOperation, BuildOperation, SummaryOperation
from ..constants import BackboneOperation


class BackboneOperationManager:
    """Manager for coordinating backbone operations with UI feedback."""
    
    def __init__(self, operation_container=None):
        """
        Initialize operation manager.
        
        Args:
            operation_container: UI operation container for progress and logging
        """
        self.logger = get_module_logger('backbone.operation_manager')
        self.operation_container = operation_container
        
        # Initialize operation handlers
        self.operations = {
            BackboneOperation.VALIDATE.value: ValidateOperation(operation_container),
            BackboneOperation.LOAD.value: LoadOperation(operation_container),
            BackboneOperation.BUILD.value: BuildOperation(operation_container),
            BackboneOperation.SUMMARY.value: SummaryOperation(operation_container)
        }
        
        self.logger.info("🎯 BackboneOperationManager initialized")
    
    async def execute_operation(self, 
                              operation_type: str, 
                              config: Dict[str, Any],
                              progress_callback: Optional[Callable] = None,
                              log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute a backbone operation.
        
        Args:
            operation_type: Type of operation to execute
            config: Configuration for the operation
            progress_callback: Optional progress callback
            log_callback: Optional log callback
            
        Returns:
            Dict containing operation results
        """
        if operation_type not in self.operations:
            error_msg = f"Unknown operation type: {operation_type}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        
        try:
            operation = self.operations[operation_type]
            
            # Execute based on operation type
            if operation_type == BackboneOperation.VALIDATE.value:
                return await operation.validate_config(config, progress_callback, log_callback)
            elif operation_type == BackboneOperation.LOAD.value:
                return await operation.load_model(config, progress_callback, log_callback)
            elif operation_type == BackboneOperation.BUILD.value:
                return await operation.build_architecture(config, progress_callback, log_callback)
            elif operation_type == BackboneOperation.SUMMARY.value:
                return await operation.generate_summary(config, progress_callback, log_callback)
            else:
                return {
                    'success': False,
                    'error': f"Operation {operation_type} not implemented"
                }
                
        except Exception as e:
            self.logger.error(f"Operation {operation_type} failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_available_operations(self) -> Dict[str, str]:
        """Get list of available operations."""
        return {
            BackboneOperation.VALIDATE.value: "Validate backbone configuration",
            BackboneOperation.LOAD.value: "Load backbone model",
            BackboneOperation.BUILD.value: "Build backbone architecture", 
            BackboneOperation.SUMMARY.value: "Generate model summary"
        }
    
    def is_operation_running(self, operation_type: str) -> bool:
        """Check if an operation is currently running."""
        if operation_type in self.operations:
            return self.operations[operation_type].is_operation_running()
        return False
    
    def cancel_operation(self, operation_type: str) -> bool:
        """Cancel a running operation."""
        if operation_type in self.operations:
            return self.operations[operation_type].cancel_operation()
        return False
    
    def shutdown(self):
        """Shutdown all operations."""
        for operation in self.operations.values():
            try:
                operation.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down operation: {e}")
        
        self.logger.info("🎯 BackboneOperationManager shutdown complete")