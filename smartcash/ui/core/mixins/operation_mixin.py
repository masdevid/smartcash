"""
Operation handling mixin for UI modules.

Provides standard operation execution, error handling, and result formatting.
"""

from typing import Dict, Any, Optional, Callable
from functools import wraps
# Removed problematic imports for now


def operation_handler(operation_name: str = None, requires_initialization: bool = True):
    """
    Decorator for UI module operations.
    
    Args:
        operation_name: Name of the operation for logging
        requires_initialization: Whether the module must be initialized
        
    Returns:
        Decorated function with standard operation handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            op_name = operation_name or func.__name__
            
            try:
                # Check initialization if required
                if requires_initialization:
                    if not getattr(self, '_is_initialized', False):
                        if not self.initialize():
                            return {
                                'success': False, 
                                'message': 'Module not initialized',
                                'operation': op_name
                            }
                
                # Execute operation
                result = func(self, *args, **kwargs)
                
                # Ensure result is in standard format
                if isinstance(result, dict):
                    result.setdefault('success', True)
                    result.setdefault('operation', op_name)
                    return result
                else:
                    return {
                        'success': True,
                        'data': result,
                        'operation': op_name,
                        'message': f'{op_name} completed successfully'
                    }
                    
            except Exception as e:
                error_msg = f"{op_name} failed: {str(e)}"
                if hasattr(self, 'logger'):
                    self.logger.error(error_msg, exc_info=True)
                
                return {
                    'success': False,
                    'message': error_msg,
                    'operation': op_name,
                    'error': str(e)
                }
        
        return wrapper
    return decorator


class OperationMixin:
    """
    Mixin providing common operation handling functionality.
    
    This mixin provides:
    - Standard operation execution patterns
    - Error handling and result formatting
    - Operation manager integration
    - Progress tracking during operations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation_manager: Optional[Any] = None
        self._operation_handlers: Dict[str, Callable] = {}
        self._is_initialized: bool = False
    
    def register_operation_handler(self, name: str, handler: Callable) -> None:
        """
        Register an operation handler.
        
        Args:
            name: Operation name
            handler: Handler function
        """
        self._operation_handlers[name] = handler
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"🔧 Registered operation handler: {name}")
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a named operation.
        
        Args:
            operation_name: Name of operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result dictionary
        """
        if operation_name not in self._operation_handlers:
            return {
                'success': False,
                'message': f'Unknown operation: {operation_name}',
                'operation': operation_name
            }
        
        try:
            handler = self._operation_handlers[operation_name]
            result = handler(*args, **kwargs)
            
            # Ensure standard format
            if isinstance(result, dict):
                result.setdefault('success', True)
                result.setdefault('operation', operation_name)
                return result
            else:
                return {
                    'success': True,
                    'data': result,
                    'operation': operation_name
                }
                
        except Exception as e:
            error_msg = f"Operation {operation_name} failed: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'message': error_msg,
                'operation': operation_name,
                'error': str(e)
            }
    
    def list_operations(self) -> Dict[str, str]:
        """
        List all registered operations.
        
        Returns:
            Dictionary of operation_name -> description
        """
        return {
            name: getattr(handler, '__doc__', f'Operation: {name}')
            for name, handler in self._operation_handlers.items()
        }
    
    @operation_handler('initialize', requires_initialization=False)
    def initialize(self) -> bool:
        """
        Initialize the module.
        
        Returns:
            True if initialization was successful
        """
        try:
            if self._is_initialized:
                return True
            
            # Initialize config handler if available
            if hasattr(self, '_initialize_config_handler'):
                self._initialize_config_handler()
            
            # Initialize operation manager if available
            if hasattr(self, '_initialize_operation_manager'):
                self._initialize_operation_manager()
            
            # Setup UI components if available
            if hasattr(self, '_setup_ui_components'):
                self._setup_ui_components()
            
            # Setup button handlers if available
            if hasattr(self, '_setup_button_handlers'):
                self._setup_button_handlers()
            
            self._is_initialized = True
            
            if hasattr(self, 'logger'):
                self.logger.info("✅ Module initialized successfully")
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize module: {e}")
            return False
    
    @operation_handler('get_status')
    def get_status(self) -> Dict[str, Any]:
        """
        Get module status information.
        
        Returns:
            Status information dictionary
        """
        return {
            'initialized': self._is_initialized,
            'has_config_handler': self._config_handler is not None,
            'has_operation_manager': self._operation_manager is not None,
            'has_ui_components': getattr(self, '_ui_components', None) is not None,
            'registered_operations': list(self._operation_handlers.keys())
        }
    
    @operation_handler('cleanup')
    def cleanup(self) -> None:
        """Clean up module resources."""
        try:
            # Cleanup operation manager
            if self._operation_manager and hasattr(self._operation_manager, 'cleanup'):
                self._operation_manager.cleanup()
            
            # Cleanup config handler
            if hasattr(self, '_config_handler') and self._config_handler:
                if hasattr(self._config_handler, 'cleanup'):
                    self._config_handler.cleanup()
            
            # Clear UI components
            if hasattr(self, '_ui_components'):
                self._ui_components = None
            
            # Clear handlers
            self._operation_handlers.clear()
            
            self._is_initialized = False
            
            if hasattr(self, 'logger'):
                self.logger.debug("🧹 Module cleaned up")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup: {e}")
    
    def update_operation_status(self, message: str, level: str = 'info') -> None:
        """
        Update operation status display.
        
        Args:
            message: Status message
            level: Message level (info, warning, error)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'update_status'):
                self._operation_manager.update_status(message, level)
                return
            
            # CORRECT: Status updates go to header_container, NOT operation_container
            if hasattr(self, '_ui_components') and self._ui_components:
                header_container = self._ui_components.get('header_container')
                if header_container and hasattr(header_container, 'update_status'):
                    header_container.update_status(message, level)
                    return
            
            # During initialization, suppress status logging to avoid console spam
            # Status updates during init are not critical for display
            if hasattr(self, '_is_initialized') and not self._is_initialized:
                # Module is still initializing - suppress status logs
                return
            
            # Only log if module is fully initialized and no other options worked
            if hasattr(self, 'log') and hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                self.log(f"[Status] {message}", level)
            elif hasattr(self, 'logger'):
                # Use debug level to avoid console spam
                self.logger.debug(f"[Status] {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update operation status: {e}")
    
    def update_progress(self, progress: int, message: str = "", level: str = "primary") -> None:
        """
        Update progress display - delegates to operation_container.
        
        Progress updates go to operation_container for centralized handling.
        
        Args:
            progress: Progress value (0-100)
            message: Progress message
            level: Progress level (primary, secondary, tertiary)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'update_progress'):
                self._operation_manager.update_progress(progress, message, level)
                return
            
            # Delegate to operation_container for progress updates
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'update_progress'):
                    operation_container.update_progress(progress, message, level)
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"[Progress] {progress}% - {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update progress: {e}")
    
    def log_operation(self, message: str, level: str = "info") -> None:
        """
        Log operation message - delegates to operation_container.
        
        Log messages go to operation_container for centralized handling.
        
        Args:
            message: Log message
            level: Log level (info, warning, error)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'log'):
                self._operation_manager.log(message, level)
                return
            
            # Delegate to operation_container for log messages
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'log'):
                    operation_container.log(message, level)
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                getattr(self.logger, level, self.logger.info)(message)
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to log operation message: {e}")
    
    def get_operation_result(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a previous operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation result or None if not found
        """
        if self._operation_manager and hasattr(self._operation_manager, 'get_result'):
            return self._operation_manager.get_result(operation_name)
        
        return None
    
    def is_operation_running(self, operation_name: str = None) -> bool:
        """
        Check if an operation is currently running.
        
        Args:
            operation_name: Specific operation name (None for any)
            
        Returns:
            True if operation is running
        """
        if self._operation_manager and hasattr(self._operation_manager, 'is_running'):
            return self._operation_manager.is_running(operation_name)
        
        return False
    
    def update_summary(self, content: str, theme: str = "default") -> None:
        """
        Update summary container content.
        
        Args:
            content: HTML content to display in summary
            theme: Theme for the summary container
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'update_summary'):
                self._operation_manager.update_summary(content, theme)
                return
            
            # Delegate to summary_container for summary updates
            if hasattr(self, '_ui_components') and self._ui_components:
                summary_container = self._ui_components.get('summary_container')
                if summary_container and hasattr(summary_container, 'set_html'):
                    summary_container.set_html(content, theme)
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"[Summary] {content}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update summary: {e}")
    
    def show_summary_message(self, title: str, message: str, message_type: str = "info", icon: str = None) -> None:
        """
        Show a message in the summary container.
        
        Args:
            title: Message title
            message: Message content
            message_type: Message type (info, success, warning, danger, primary)
            icon: Optional icon to display
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'show_summary_message'):
                self._operation_manager.show_summary_message(title, message, message_type, icon)
                return
            
            # Delegate to summary_container for message display
            if hasattr(self, '_ui_components') and self._ui_components:
                summary_container = self._ui_components.get('summary_container')
                if summary_container and hasattr(summary_container, 'show_message'):
                    summary_container.show_message(title, message, message_type, icon)
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.info(f"[Summary] {title}: {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to show summary message: {e}")
    
    def show_summary_status(self, items: Dict[str, Any], title: str = "", icon: str = "") -> None:
        """
        Show status items in the summary container.
        
        Args:
            items: Dictionary of status items and their values
            title: Optional title for the status
            icon: Optional icon for the title
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'show_summary_status'):
                self._operation_manager.show_summary_status(items, title, icon)
                return
            
            # Delegate to summary_container for status display
            if hasattr(self, '_ui_components') and self._ui_components:
                summary_container = self._ui_components.get('summary_container')
                if summary_container and hasattr(summary_container, 'show_status'):
                    summary_container.show_status(items, title, icon)
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                status_text = ", ".join([f"{k}: {v}" for k, v in items.items()])
                self.logger.info(f"[Summary Status] {title}: {status_text}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to show summary status: {e}")
    
    def clear_summary(self) -> None:
        """Clear the summary container content."""
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'clear_summary'):
                self._operation_manager.clear_summary()
                return
            
            # Delegate to summary_container for clearing
            if hasattr(self, '_ui_components') and self._ui_components:
                summary_container = self._ui_components.get('summary_container')
                if summary_container and hasattr(summary_container, 'clear'):
                    summary_container.clear()
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug("[Summary] Cleared")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to clear summary: {e}")
    
    def show_operation_dialog(self, message: str, title: str = "Operation", dialog_type: str = "info", buttons: Optional[Dict[str, Any]] = None, callback: Optional[callable] = None) -> None:
        """
        Show an operation dialog.
        
        Args:
            message: Dialog message
            title: Dialog title
            dialog_type: Dialog type (info, confirmation, warning, error)
            buttons: Optional custom buttons configuration
            callback: Optional callback function for dialog actions
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'show_operation_dialog'):
                self._operation_manager.show_operation_dialog(message, title, dialog_type, buttons, callback)
                return
            
            # Delegate to operation_container for dialog display
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'show_dialog'):
                    operation_container.show_dialog(message, title, dialog_type, buttons, callback)
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.info(f"[Dialog] {title}: {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to show operation dialog: {e}")
    
    def clear_operation_dialog(self) -> None:
        """Clear the operation dialog."""
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'clear_operation_dialog'):
                self._operation_manager.clear_operation_dialog()
                return
            
            # Delegate to operation_container for dialog clearing
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'clear_dialog'):
                    operation_container.clear_dialog()
                    return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug("[Dialog] Cleared")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to clear operation dialog: {e}")