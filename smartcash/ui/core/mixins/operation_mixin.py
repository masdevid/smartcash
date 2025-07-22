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
    
    Note: Status updates have been moved to LoggingMixin as they are more
    closely related to logging functionality.
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
    
    # Logging functionality is now handled through the standard log method
    # which provides consistent logging behavior across the application.
    
    def update_progress(self, progress: int, message: str = "", level: str = "primary", 
                       secondary_progress: Optional[int] = None, secondary_message: str = "") -> None:
        """
        Update progress display - delegates to operation_container.
        
        Progress updates go to operation_container for centralized handling.
        Supports both single and dual progress tracking.
        
        Args:
            progress: Progress value (0-100)
            message: Progress message
            level: Progress level (primary, secondary, tertiary)
            secondary_progress: Optional secondary progress value (0-100) for dual progress
            secondary_message: Optional secondary progress message for dual progress
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'update_progress'):
                self._operation_manager.update_progress(progress, message, level)
                return
            
            # Delegate to operation_container for progress updates
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'update_progress'):
                        operation_container.update_progress(progress, message, level)
                        # Handle secondary progress if provided
                        if secondary_progress is not None:
                            operation_container.update_progress(secondary_progress, secondary_message, 'secondary')
                        return
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict) and 'update_progress' in operation_container:
                        operation_container['update_progress'](progress, message, level)
                        # Handle secondary progress if provided
                        if secondary_progress is not None:
                            operation_container['update_progress'](secondary_progress, secondary_message, 'secondary')
                        return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"[Progress] {progress}% - {message}")
                if secondary_progress is not None:
                    self.logger.debug(f"[Secondary Progress] {secondary_progress}% - {secondary_message}")
                
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
    
    def show_operation_dialog(self, title: str, message: str, on_confirm: Optional[Callable] = None, on_cancel: Optional[Callable] = None, confirm_text: str = "Confirm", cancel_text: str = "Cancel", danger_mode: bool = False) -> None:
        """
        Show an operation dialog (confirmation dialog).
        
        Args:
            title: Dialog title
            message: Dialog message
            on_confirm: Callback when user confirms
            on_cancel: Callback when user cancels  
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            danger_mode: If True, shows the confirm button in danger color
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'show_operation_dialog'):
                self._operation_manager.show_operation_dialog(title, message, on_confirm, on_cancel, confirm_text, cancel_text, danger_mode)
                return
            
            # Delegate to operation_container for dialog display
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'show_dialog'):
                        operation_container.show_dialog(title, message, on_confirm, on_cancel, confirm_text, cancel_text, danger_mode)
                        return
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict) and 'show_dialog' in operation_container:
                        operation_container['show_dialog'](title, message, on_confirm, on_cancel, confirm_text, cancel_text, danger_mode)
                        return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.info(f"[Dialog] {title}: {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to show operation dialog: {e}")
    
    def show_info_dialog(self, title: str, message: str, on_ok: Optional[Callable] = None, ok_text: str = "OK", info_type: str = "info") -> None:
        """
        Show an info dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback when user clicks OK
            ok_text: Text for OK button
            info_type: Type of info dialog (info, success, warning, error)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'show_info_dialog'):
                self._operation_manager.show_info_dialog(title, message, on_ok, ok_text, info_type)
                return
            
            # Delegate to operation_container for info dialog display
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'show_info'):
                        operation_container.show_info(title, message, on_ok, ok_text, info_type)
                        return
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict) and 'show_info' in operation_container:
                        operation_container['show_info'](title, message, on_ok, ok_text, info_type)
                        return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.info(f"[Info Dialog] {title}: {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to show info dialog: {e}")
    
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
    
    def start_progress(self, message: str = "Starting...", progress: int = 0, level: str = "primary") -> None:
        """
        Start progress tracking - initializes progress display for an operation.
        
        This method is semantically different from update_progress() as it:
        - Initializes progress tracking for a new operation
        - Typically sets progress to 0% 
        - May reset previous progress states
        
        Args:
            message: Initial progress message
            progress: Initial progress value (0-100, typically 0)
            level: Progress level (primary, secondary, tertiary)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'start_progress'):
                self._operation_manager.start_progress(message, progress, level)
                return
            
            # Delegate to operation_container - use update_progress since no dedicated start method exists
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'start_progress'):
                        operation_container.start_progress(message, progress, level)
                        return
                    elif hasattr(operation_container, 'update_progress'):
                        operation_container.update_progress(progress, message, level)
                        return
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict):
                        if 'start_progress' in operation_container:
                            operation_container['start_progress'](message, progress, level)
                            return
                        elif 'update_progress' in operation_container:
                            operation_container['update_progress'](progress, message, level)
                            return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"[Progress Start] {progress}% - {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to start progress: {e}")
    
    def complete_progress(self, message: str = "Completed!", level: str = "primary") -> None:
        """
        Complete progress tracking - delegates to operation_container.
        
        Args:
            message: Completion message
            level: Progress level (primary, secondary, tertiary)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'complete_progress'):
                self._operation_manager.complete_progress(message, level)
                return
            
            # Delegate to operation_container for progress completion
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'complete_progress'):
                        operation_container.complete_progress(message, level)
                        return
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict) and 'update_progress' in operation_container:
                        # Use update_progress with 100% for completion
                        operation_container['update_progress'](100, message, level)
                        return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"[Progress Complete] {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to complete progress: {e}")
    
    def error_progress(self, message: str = "An error occurred!", level: str = "primary") -> None:
        """
        Set progress to error state - delegates to operation_container.
        
        Args:
            message: Error message
            level: Progress level (primary, secondary, tertiary)
        """
        try:
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'error_progress'):
                self._operation_manager.error_progress(message, level)
                return
            
            # Delegate to operation_container for progress error
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'error_progress'):
                        operation_container.error_progress(message, level)
                        return
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict) and 'update_progress' in operation_container:
                        # Use update_progress with error indication
                        operation_container['update_progress'](0, message, 'danger')
                        return
            
            # Fallback logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"[Progress Error] {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to set error progress: {e}")
    
    # Operation logging is now handled by LoggingMixin to avoid conflicts
    # OperationMixin delegates to LoggingMixin for all logging operations
    
    def ensure_progress_ready(self) -> bool:
        """
        Ensure progress tracking components are ready.
        
        Returns:
            True if progress tracking is ready, False otherwise
        """
        try:
            # Check if operation_container is available for progress tracking
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle operation_container as OperationContainer object
                    if hasattr(operation_container, 'update_progress'):
                        return True
                    # Handle operation_container as dict (returned by create_operation_container)
                    elif isinstance(operation_container, dict) and 'update_progress' in operation_container:
                        return True
            
            # Check if operation_manager is available for progress tracking
            if self._operation_manager and hasattr(self._operation_manager, 'update_progress'):
                return True
            
            # Progress tracking not available
            return False
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to check progress readiness: {e}")
            return False