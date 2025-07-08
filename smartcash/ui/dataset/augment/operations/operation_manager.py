"""
File: smartcash/ui/dataset/augment/operations/operation_manager.py
Description: Operation manager for augment module following core patterns

This manager coordinates all augmentation operations and provides centralized
operation handling with preserved business logic.
"""

from typing import Dict, Any, Optional, List
import logging
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from ..constants import AugmentationOperation, ProcessingPhase, SUCCESS_MESSAGES, ERROR_MESSAGES


class AugmentOperationManager(BaseHandler):
    """
    Operation manager for augment module with core inheritance.
    
    Features:
    - 🏗️ Centralized operation coordination
    - 🔄 Operation state management
    - ✅ Progress tracking and status updates
    - 📊 Result aggregation
    - 🎨 Preserved business logic
    """
    
    @handle_ui_errors(error_component_title="Operation Manager Initialization Error")
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """
        Initialize operation manager.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(
            module_name='augment_operations',
            parent_module='augment',
            ui_components=ui_components
        )
        
        # Operation registry
        self._operations: Dict[str, Any] = {}
        self._current_operation: Optional[str] = None
        self._operation_history: List[Dict[str, Any]] = []
        
        # Initialize available operations
        self._initialize_operations()
        
        self.logger.info("🔧 AugmentOperationManager initialized")
    
    def _initialize_operations(self) -> None:
        """Initialize available operations."""
        # Import operation classes
        from .augment_operation import AugmentOperation
        from .check_operation import CheckOperation
        from .cleanup_operation import CleanupOperation
        from .preview_operation import PreviewOperation
        
        # Register operations
        self._operations = {
            AugmentationOperation.AUGMENT.value: AugmentOperation(self.ui_components),
            AugmentationOperation.CHECK.value: CheckOperation(self.ui_components),
            AugmentationOperation.CLEANUP.value: CleanupOperation(self.ui_components),
            AugmentationOperation.PREVIEW.value: PreviewOperation(self.ui_components)
        }
        
        self.logger.debug(f"📋 Registered {len(self._operations)} operations")
    
    @handle_ui_errors(error_component_title="Operation Execution Error")
    def execute_operation(self, operation_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specified operation with configuration.
        
        Args:
            operation_type: Type of operation to execute
            config: Operation configuration
            
        Returns:
            Dictionary containing operation results
        """
        if operation_type not in self._operations:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        self.logger.info(f"🚀 Executing {operation_type} operation")
        
        try:
            # Set current operation
            self._current_operation = operation_type
            
            # Update UI status
            self._update_operation_status(operation_type, "starting")
            
            # Execute operation
            operation = self._operations[operation_type]
            result = operation.execute(config)
            
            # Record operation history
            self._record_operation(operation_type, result)
            
            # Update UI status
            self._update_operation_status(operation_type, "completed", result)
            
            self.logger.info(f"✅ {operation_type} operation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {operation_type} operation failed: {e}")
            self._update_operation_status(operation_type, "failed", {"error": str(e)})
            raise
        finally:
            self._current_operation = None
    
    def get_operation_status(self, operation_type: str) -> Dict[str, Any]:
        """
        Get status of specified operation.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Dictionary containing operation status
        """
        if operation_type not in self._operations:
            return {"status": "unknown", "message": "Operation not found"}
        
        operation = self._operations[operation_type]
        return operation.get_status()
    
    def get_available_operations(self) -> List[str]:
        """
        Get list of available operations.
        
        Returns:
            List of available operation types
        """
        return list(self._operations.keys())
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Get operation execution history.
        
        Returns:
            List of operation history records
        """
        return self._operation_history.copy()
    
    def cancel_current_operation(self) -> bool:
        """
        Cancel currently running operation.
        
        Returns:
            True if operation was cancelled, False otherwise
        """
        if not self._current_operation:
            return False
        
        try:
            operation = self._operations[self._current_operation]
            if hasattr(operation, 'cancel'):
                operation.cancel()
                self.logger.info(f"🛑 Cancelled {self._current_operation} operation")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to cancel operation: {e}")
        
        return False
    
    def _update_operation_status(self, operation_type: str, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Update operation status in UI."""
        if not self.ui_components:
            return
        
        # Update operation summary if available
        update_methods = self.ui_components.get('update_methods', {})
        
        # Update status
        if 'status' in update_methods:
            if status == "starting":
                update_methods['status']('processing', f"Starting {operation_type}...")
            elif status == "completed":
                success_msg = SUCCESS_MESSAGES.get(f'{operation_type}_complete', f'{operation_type} completed')
                update_methods['status']('success', success_msg)
            elif status == "failed":
                error_msg = ERROR_MESSAGES.get(f'{operation_type}_failed', f'{operation_type} failed')
                update_methods['status']('error', error_msg)
        
        # Update activity log
        if 'activity' in update_methods:
            if status == "starting":
                update_methods['activity'](f"Started {operation_type} operation")
            elif status == "completed":
                update_methods['activity'](f"Completed {operation_type} operation")
            elif status == "failed":
                error_msg = result.get('error', 'Unknown error') if result else 'Unknown error'
                update_methods['activity'](f"Failed {operation_type}: {error_msg}")
    
    def _record_operation(self, operation_type: str, result: Dict[str, Any]) -> None:
        """Record operation in history."""
        import datetime
        
        record = {
            'operation_type': operation_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'result': result,
            'status': 'success' if result.get('success', False) else 'failed'
        }
        
        self._operation_history.append(record)
        
        # Keep only last 10 operations
        if len(self._operation_history) > 10:
            self._operation_history.pop(0)
        
        self.logger.debug(f"📝 Recorded {operation_type} operation in history")