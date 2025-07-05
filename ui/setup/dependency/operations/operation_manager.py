"""
File: smartcash/ui/setup/dependency/operations/operation_manager.py
Deskripsi: Manages package operations with proper state and error handling.
"""
from typing import Dict, Any, List, Optional, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import importlib

from .base_operation import BaseOperationHandler
from .install_operation import InstallOperationHandler
from .update_operation import UpdateOperationHandler
from .uninstall_operation import UninstallOperationHandler
from .check_status_operation import CheckStatusOperationHandler


class OperationType(Enum):
    """Supported operation types."""
    INSTALL = auto()
    UPDATE = auto()
    UNINSTALL = auto()
    CHECK_STATUS = auto()


@dataclass
class OperationContext:
    """Context for an operation execution."""
    operation_type: OperationType
    packages: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    status_callback: Optional[Callable[[str, str], None]] = None
    progress_callback: Optional[Callable[[int, int], None]] = None


class OperationManager:
    """Manages package operations with proper state and error handling."""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        """Initialize the operation manager.
        
        Args:
            ui_components: Dictionary of UI components for operation feedback.
        """
        self.ui_components = ui_components or {}
        self._current_operation: Optional[OperationContext] = None
        self._operation_handlers: Dict[OperationType, Type[BaseOperationHandler]] = {}
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup operation handlers."""
        self._operation_handlers = {
            OperationType.INSTALL: InstallOperationHandler,
            OperationType.UPDATE: UpdateOperationHandler,
            OperationType.UNINSTALL: UninstallOperationHandler,
            OperationType.CHECK_STATUS: CheckStatusOperationHandler
        }
    
    def create_operation_context(
        self,
        operation_type: OperationType,
        packages: List[str],
        status_callback: Optional[Callable[[str, str], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> OperationContext:
        """Create a new operation context.
        
        Args:
            operation_type: Type of operation to perform.
            packages: List of packages to operate on.
            status_callback: Optional callback for status updates.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            OperationContext: The created operation context.
        """
        requires_confirmation = operation_type in [OperationType.UNINSTALL, OperationType.UPDATE]
        return OperationContext(
            operation_type=operation_type,
            packages=packages,
            requires_confirmation=requires_confirmation,
            status_callback=status_callback,
            progress_callback=progress_callback
        )
    
    async def execute_operation(self, context: OperationContext) -> Dict[str, Any]:
        """Execute an operation with the given context.
        
        Args:
            context: The operation context.
            
        Returns:
            Dict[str, Any]: The operation result.
        """
        self._current_operation = context
        
        try:
            # Get the appropriate handler class
            handler_class = self._operation_handlers.get(context.operation_type)
            if not handler_class:
                raise ValueError(f"No handler found for operation: {context.operation_type}")
            
            # Create handler instance with UI components and config
            handler = handler_class(
                ui_components=self.ui_components,
                config={}  # Pass any required config here
            )
            
            # Update status
            self._update_status(f"🚀 Starting {context.operation_type.name.lower()}...")
            
            # Execute the operation asynchronously
            result = await handler.execute_operation()
            
            # Update status based on result
            if result.get('success', False):
                self._update_status(f"✅ {context.operation_type.name.capitalize()} completed", "success")
            else:
                error_msg = result.get('error', 'Unknown error')
                self._update_status(f"❌ {context.operation_type.name.capitalize()} failed: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            self._update_status(f"❌ {context.operation_type.name.capitalize()} failed: {str(e)}", "error")
            raise
        finally:
            self._current_operation = None
    
    def _update_status(self, message: str, level: str = "info") -> None:
        """Update operation status.
        
        Args:
            message: Status message.
            level: Message level (info, warning, error, success).
        """
        if self._current_operation and self._current_operation.status_callback:
            self._current_operation.status_callback(message, level)
        elif 'status_label' in self.ui_components:
            self.ui_components['status_label'].value = message
            
    def get_current_operation(self) -> Optional[OperationContext]:
        """Get the current operation context.
        
        Returns:
            Optional[OperationContext]: The current operation context, or None if no operation is in progress.
        """
        return self._current_operation
