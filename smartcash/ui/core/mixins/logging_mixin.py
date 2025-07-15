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
            
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'log'):
                self._operation_manager.log(message, level)
                return
            
            # Try operation container directly
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'log'):
                    operation_container.log(message, level)
                    return
            
            # Fallback to standard logger (use debug to minimize console output)
            if hasattr(self, 'logger'):
                self.logger.debug(message)
                
        except Exception as e:
            # Final fallback (use debug to minimize console output)
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to log message: {e}")
                self.logger.debug(f"[{level.upper()}] {message}")
            else:
                # Suppress print during normal operation to avoid console spam
                pass
    
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