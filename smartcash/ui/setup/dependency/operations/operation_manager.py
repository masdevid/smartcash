"""
File: smartcash/ui/setup/dependency/operations/operation_manager.py
Deskripsi: Manages package operations with proper state and error handling.
"""
import asyncio
from typing import Dict, Any, List, Optional, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import importlib

from .base_operation import BaseOperationHandler
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from .install_operation import InstallOperationHandler
from .update_operation import UpdateOperationHandler
from .uninstall_operation import UninstallOperationHandler
from .check_operation import CheckStatusOperationHandler


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
        self._operation_lock = asyncio.Lock()  # Lock for thread-safe operation execution
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
            
        Raises:
            RuntimeError: If another operation is already in progress.
        """
        # Check if an operation is already in progress
        if not self._operation_lock.locked():
            async with self._operation_lock:
                self._current_operation = context
                return await self._execute_operation(context)
        else:
            return {
                'success': False,
                'error': 'Another operation is already in progress',
                'message': 'Cannot start a new operation while another is in progress'
            }
        
    async def _execute_operation(self, context: OperationContext) -> Dict[str, Any]:
        """Internal method to execute an operation.
        
        Args:
            context: The operation context.
            
        Returns:
            Dict[str, Any]: The operation result.
            
        Raises:
            ValueError: If no packages are provided for the operation.
        """
        # Validate packages
        if not context.packages:
            return {
                'success': False,
                'error': 'No packages provided',
                'message': 'No packages specified for operation'
            }
            
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
            result = await handler.execute_operation(context)
            
            # Update status based on result
            if result.get('success', False):
                self._update_status(f"✅ {context.operation_type.name.capitalize()} completed", "success")
            else:
                error_msg = result.get('error', 'Unknown error')
                self._update_status(f"❌ {context.operation_type.name.capitalize()} failed: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            self._update_status(f"❌ {context.operation_type.name.capitalize()} failed: {str(e)}", "error")
            return {
                'success': False,
                'error': str(e),
                'message': f"Operation failed: {str(e)}"
            }
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


class DependencyOperationManager(OperationHandler):
    """Operation manager for dependency management that extends OperationHandler."""
    
    def __init__(self, config: Dict[str, Any], operation_container=None, **kwargs):
        """Initialize the dependency operation manager."""
        super().__init__(
            module_name='dependency_operation_manager',
            parent_module='dependency',
            operation_container=operation_container,
            **kwargs
        )
        self.config = config
        self.operation_manager = OperationManager(ui_components={})
    
    def initialize(self) -> None:
        """Initialize the dependency operation manager."""
        self.logger.info("🚀 Initializing Dependency operation manager")
        # No specific initialization needed
        self.logger.info("✅ Dependency operation manager initialization complete")
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'install': self.execute_install,
            'uninstall': self.execute_uninstall, 
            'update': self.execute_update,
            'check_status': self.execute_check_status
        }
    
    async def execute_install(self, packages: List[str], progress_callback=None) -> Dict[str, Any]:
        """Execute package installation using real pip install."""
        try:
            from .install_operation import InstallOperationHandler
            
            # Create install handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
                
            install_handler = InstallOperationHandler(ui_components, self.config)
            
            # Execute the actual installation
            result = await install_handler.execute_operation()
            
            # Update operation summary with results
            await self._update_operation_summary('install', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_install: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_uninstall(self, packages: List[str], progress_callback=None) -> Dict[str, Any]:
        """Execute package uninstallation using real pip uninstall."""
        try:
            from .uninstall_operation import UninstallOperationHandler
            
            # Create uninstall handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
                
            uninstall_handler = UninstallOperationHandler(ui_components, self.config)
            
            # Execute the actual uninstallation
            result = await uninstall_handler.execute_operation()
            
            # Update operation summary with results
            await self._update_operation_summary('uninstall', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_uninstall: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_update(self, packages: List[str], progress_callback=None) -> Dict[str, Any]:
        """Execute package update using real pip install --upgrade."""
        try:
            from .update_operation import UpdateOperationHandler
            
            # Create update handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
                
            update_handler = UpdateOperationHandler(ui_components, self.config)
            
            # Execute the actual update
            result = await update_handler.execute_operation()
            
            # Update operation summary with results
            await self._update_operation_summary('update', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_update: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_check_status(self, packages: List[str] = None, progress_callback=None) -> Dict[str, Any]:
        """Execute package status check using real pip show."""
        try:
            from .check_status_operation import CheckStatusOperationHandler
            
            # Create check status handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
                
            check_handler = CheckStatusOperationHandler(ui_components, self.config)
            
            # Execute the actual status check
            result = await check_handler.execute_operation()
            
            # Update operation summary with results
            await self._update_operation_summary('check_status', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_check_status: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _update_operation_summary(self, operation_type: str, result: Dict[str, Any]) -> None:
        """Update operation summary with operation results."""
        try:
            from ..components.operation_summary import update_operation_summary
            
            # Get the operation summary widget from UI components
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            operation_summary = ui_components.get('operation_summary')
            if operation_summary:
                # Determine status type based on result
                if result.get('cancelled'):
                    status_type = 'warning'
                elif result.get('success'):
                    status_type = 'success'
                else:
                    status_type = 'error'
                
                # Update the summary widget
                update_operation_summary(operation_summary, operation_type, result, status_type)
                self.logger.info(f"✅ Updated operation summary for {operation_type}")
            else:
                self.logger.warning("⚠️ Operation summary widget not found in UI components")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update operation summary: {e}")
