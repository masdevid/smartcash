"""
Logging mixin for UI modules.

Provides standard logging functionality with operation container integration.
"""

from typing import Dict, Any, Optional
# Removed problematic import for now


class LoggingMixin:
    """
    Mixin providing common logging functionality.
    
    This mixin provides:
    - Operation container logging integration
    - Fallback to standard logger
    - UI logging bridge setup
    - Standard log message formatting
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ui_logging_bridge_setup: bool = False
        self._log_buffer: list = []  # Buffer logs until operation container is ready
    
    def _get_module_namespace(self) -> str:
        """
        Get the appropriate namespace for this module's logs.
        
        Returns:
            Namespace string that matches the log_namespace_filter
        """
        # Check if module has explicit namespace info
        if hasattr(self, 'module_name'):
            return self.module_name
        
        # Check if this is a BaseUIModule with module_name
        if hasattr(self, 'full_module_name'):
            return self.full_module_name
        
        # Try to infer from class name
        class_name = self.__class__.__name__.lower()
        if 'colab' in class_name:
            return 'colab'
        elif 'downloader' in class_name:
            return 'downloader'
        elif 'split' in class_name:
            return 'split'
        elif 'preprocess' in class_name:
            return 'preprocess'
        elif 'dependency' in class_name:
            return 'dependency'
        elif 'augment' in class_name:
            return 'augment'
        
        # Default fallback
        return 'smartcash.ui.core'
    
    def log(self, message: str, level: str = 'info') -> None:
        """
        Log message to operation container or fallback to logger.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        try:
            # Check if we should buffer logs (operation container not ready yet)
            if hasattr(self, '_log_buffer') and hasattr(self, '_is_initialized'):
                operation_container = None
                
                # Check for operation container availability
                if hasattr(self, '_ui_components') and self._ui_components:
                    operation_container = self._ui_components.get('operation_container')
                
                # If operation container isn't ready but we're initialized, buffer the log
                if not operation_container and self._is_initialized:
                    self._log_buffer.append((message, level))
                    return
            
            # Determine namespace for this module
            namespace = self._get_module_namespace()
            
            # Try operation container directly
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle dict-style operation container
                    if isinstance(operation_container, dict):
                        if 'log_message' in operation_container:
                            operation_container['log_message'](message, level, namespace)
                            return
                        elif 'log' in operation_container:  # For backward compatibility
                            operation_container['log'](message, level, namespace)
                            return
                    # Handle object-style operation container
                    elif hasattr(operation_container, 'log_message'):
                        operation_container.log_message(message, level, namespace)
                        return
                    elif hasattr(operation_container, 'log'):  # For backward compatibility
                        operation_container.log(message, level, namespace)
                        return
            
            # Fallback to standard logger (use debug to minimize console output)
            if hasattr(self, 'logger'):
                self.logger.debug(f"[{level.upper()}] {message}")
                
        except Exception as e:
            # Final fallback (use debug to minimize console output)
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to log message: {e}")
                level_str = level.name.lower() if hasattr(level, 'name') else str(level).lower()
                self.logger.debug(f"[{level_str.upper()}] {message}")
            else:
                # Suppress print during normal operation to avoid console spam
                pass
    
    def update_operation_status(self, message: str, level: str = 'info') -> None:
        """
        Update operation status display.
        
        This method is responsible for updating the UI with operation status messages.
        It's placed in LoggingMixin since status updates are closely related to logging
        and often used in conjunction with logging operations.
        
        Args:
            message: Status message to display
            level: Message level ('info', 'warning', 'error', 'success')
        """
        try:
            # Status updates go to header_container
            if hasattr(self, '_ui_components') and self._ui_components:
                header_container = self._ui_components.get('header_container')
                if header_container and hasattr(header_container, 'update_status'):
                    header_container.update_status(message, level)
                    return
            
            # Fallback to logging if header container not available
            if hasattr(self, 'log') and hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                self.log(f"[Status] {message}", level)
            elif hasattr(self, 'logger'):
                # Use debug level to avoid console spam
                self.logger.debug(f"[Status] {message}")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update operation status: {e}")
    
    def log_with_status(self, message: str, status_message: str = None, status_level: str = 'info', log_level: str = 'info') -> None:
        """
        Log a message and optionally update operation status.
        
        This is a convenience wrapper that combines logging and status updates
        in a single call when both are needed.
        
        Args:
            message: Message to log
            status_message: Optional status message (if None, same as message)
            status_level: Level for status update ('info', 'warning', 'error', 'success')
            log_level: Level for logging ('debug', 'info', 'warning', 'error')
        """
        # Always log the message
        self.log(message, log_level)
        
        # Update status if a status message is provided or if it should mirror the log message
        if status_message is not None or status_message != '':
            status_msg = status_message if status_message is not None else message
            self.update_operation_status(status_msg, status_level)
    
    def _setup_ui_logging_bridge(self, operation_container: Any) -> None:
        """
        Setup UI logging bridge to capture backend service logs.
        
        Args:
            operation_container: Operation container to bridge logs to
        """
        try:
            if self._ui_logging_bridge_setup:
                return
                
            # Setup logging bridge if operation container supports it
            if hasattr(operation_container, 'setup_logging_bridge'):
                operation_container.setup_logging_bridge()
                
            # Setup logger handlers to redirect to UI
            if hasattr(self, 'logger') and hasattr(operation_container, 'capture_logs'):
                operation_container.capture_logs(self.logger)
                
            self._ui_logging_bridge_setup = True
            
            if hasattr(self, 'logger'):
                self.logger.debug("✅ UI logging bridge setup complete")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to setup UI logging bridge: {e}")
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.log(message, 'info')
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, 'warning')
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.log(message, 'error')
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.log(message, 'debug')
    
    def log_success(self, message: str) -> None:
        """Log success message."""
        self.log(f"✅ {message}", 'info')
    
    def log_operation_start(self, operation_name: str) -> None:
        """Log operation start."""
        self.log(f"🔄 Starting {operation_name}...", 'info')
    
    def log_operation_complete(self, operation_name: str) -> None:
        """Log operation completion."""
        self.log(f"✅ {operation_name} completed", 'info')
    
    def log_operation_error(self, operation_name: str, error: str) -> None:
        """Log operation error."""
        self.log(f"❌ {operation_name} failed: {error}", 'error')
    
    def clear_logs(self) -> None:
        """Clear operation container logs."""
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'clear_logs'):
                self._operation_manager.clear_logs()
            elif hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'clear_logs'):
                    operation_container.clear_logs()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to clear logs: {e}")
    
    def get_logs(self) -> Optional[str]:
        """
        Get current logs from operation container.
        
        Returns:
            Current logs or None if not available
        """
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'get_logs'):
                return self._operation_manager.get_logs()
            elif hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'get_logs'):
                    return operation_container.get_logs()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to get logs: {e}")
        
        return None
    
    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container when it becomes available."""
        try:
            if hasattr(self, '_log_buffer') and self._log_buffer:
                # Get operation container
                operation_container = None
                if hasattr(self, '_ui_components') and self._ui_components:
                    operation_container = self._ui_components.get('operation_container')
                
                if operation_container:
                    # Get namespace for this module
                    namespace = self._get_module_namespace()
                    
                    # Flush all buffered logs
                    for message, level in self._log_buffer:
                        if isinstance(operation_container, dict) and 'log' in operation_container:
                            operation_container['log'](message, level, namespace)
                        elif hasattr(operation_container, 'log'):
                            operation_container.log(message, level, namespace)
                    
                    # Clear buffer after flushing
                    self._log_buffer.clear()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to flush log buffer: {e}")