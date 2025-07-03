# smartcash/ui/core/shared/logger.py
"""
Centralized logger for SmartCash UI components with suppression support.
Extends the existing UILogger with additional features for core components.
"""
from typing import Dict, Any, Optional, Union
import logging
import sys
from IPython.display import display

from smartcash.ui.utils.ui_logger import get_module_logger, UILogger as BaseUILogger


class UILogger(BaseUILogger):
    """
    Enhanced UILogger for core components with suppression support.
    
    This class extends the base UILogger from smartcash.ui.utils.ui_logger with
    additional features for log suppression and context management.
    
    Key features:
    - Automatically suppresses logs until log_output is ready
    - Ensures all logs are directed to log_output only
    - Provides methods for temporary log suppression
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: str = "ui",
        ui_components: Optional[Dict[str, Any]] = None,
        log_level: str = "info",
        suppress_logs: bool = None
    ):
        """
        Initialize the enhanced UI logger.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
            ui_components: Dictionary containing UI components
            log_level: Logging level (debug, info, warning, error, critical)
            suppress_logs: If True, logs will be suppressed; if None, auto-determine based on log_output readiness
        """
        # Convert string log level to int for base class
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        int_log_level = level_map.get(log_level.lower(), logging.INFO)
        
        # Initialize the base UILogger
        super().__init__(ui_components, name=f"smartcash.{parent_module}.{module_name}", log_level=int_log_level)
        
        # Store additional properties
        self.module_name = module_name
        self.parent_module = parent_module
        
        # Auto-determine suppression if not explicitly set
        if suppress_logs is None:
            # Suppress logs if log_output is not ready
            self.suppress_logs = not self._is_log_output_ready()
        else:
            self.suppress_logs = suppress_logs
    
    def _is_log_output_ready(self) -> bool:
        """
        Check if log_output is ready to receive logs.
        
        Returns:
            True if log_output is ready, False otherwise
        """
        if not self.ui_components:
            return False
            
        log_output = self.ui_components.get('log_output')
        if log_output is None:
            return False
            
        # Check if log_output has append_text method
        return hasattr(log_output, 'append_text')
    
    def _log_to_ui(self, level: str, message: str) -> None:
        """
        Override base _log_to_ui to add suppression support.
        
        Args:
            level: Log level
            message: Log message
        """
        # Skip UI logging if logs are suppressed or log_output is not ready
        if self.suppress_logs or not self._is_log_output_ready():
            return
            
        # Call the parent method to handle the actual logging
        super()._log_to_ui(level, message)
        
    def set_suppression(self, suppress: bool) -> None:
        """
        Enable or disable log suppression.
        
        Args:
            suppress: True to suppress logs, False to enable them
        """
        self.suppress_logs = suppress
        
    def update_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Update UI components and check if log_output is ready.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().update_ui_components(ui_components)
        
        # Auto-unsuppress if log_output is now ready
        if self.suppress_logs and self._is_log_output_ready():
            self.suppress_logs = False
        
    def suppress(self) -> None:
        """Enable log suppression."""
        self.suppress_logs = True
        
    def unsuppress(self) -> None:
        """Disable log suppression."""
        self.suppress_logs = False
        
    def with_suppression(self, suppress: bool = True):
        """
        Context manager for temporary log suppression.
        
        Usage:
            with logger.with_suppression():
                # Logs are suppressed in this block
            # Logs are restored to previous state outside the block
            
        Args:
            suppress: True to suppress logs, False to enable them
            
        Returns:
            Context manager for log suppression
        """
        class LogSuppressionContext:
            def __init__(self, logger, suppress):
                self.logger = logger
                self.suppress = suppress
                self.previous_state = None
                
            def __enter__(self):
                self.previous_state = self.logger.suppress_logs
                self.logger.suppress_logs = self.suppress
                return self.logger
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.logger.suppress_logs = self.previous_state
                
        return LogSuppressionContext(self, suppress)


def get_ui_logger(
    module_name: str = None,
    parent_module: str = "ui",
    ui_components: Optional[Dict[str, Any]] = None,
    log_level: str = "info",
    suppress_logs: bool = None
) -> UILogger:
    """
    Get an enhanced UI logger instance with suppression support.
    
    Args:
        module_name: Name of the module (defaults to caller's module name if None)
        parent_module: Parent module name
        ui_components: Dictionary containing UI components
        log_level: Logging level (debug, info, warning, error, critical)
        suppress_logs: If True, logs will be suppressed; if None, auto-determine based on log_output readiness
        
    Returns:
        Enhanced UILogger instance
    """
    # Use caller's module name if not provided
    if module_name is None:
        import inspect
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        if module:
            module_name = module.__name__.split('.')[-1]
        else:
            module_name = "unknown"
            
    return UILogger(module_name, parent_module, ui_components, log_level, suppress_logs)
