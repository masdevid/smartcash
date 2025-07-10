"""
Global UI Logging Manager - DRY solution for consistent logging control across all UI modules.

This module provides a centralized way to:
1. Suppress initial logs during UI initialization
2. Route logs to UI components only (not console)
3. Manage logging lifecycle consistently across all modules
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
import threading


class UILoggingManager:
    """Global singleton for managing UI logging across all modules."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._ui_handlers: Dict[str, logging.Handler] = {}
        self._original_levels: Dict[str, int] = {}
        self._suppression_active = False
        
        # Loggers that should be managed by UI logging
        self._managed_loggers = [
            'smartcash.ui',
            'smartcash.ui.setup',
            'smartcash.ui.components', 
            'smartcash.ui.core',
            'smartcash.common',
            'smartcash.dataset',
            'smartcash.preprocess'
        ]
    
    @contextmanager
    def suppress_initialization_logs(self, test_mode: bool = False):
        """Context manager to suppress logs during UI initialization.
        
        Args:
            test_mode: If True, suppress all logs. If False, allow WARNING+ through.
        """
        if self._suppression_active:
            # Already suppressed, just yield
            yield
            return
            
        self._suppression_active = True
        
        # Store original levels
        loggers_to_suppress = [
            logging.getLogger(),  # Root logger
            logging.getLogger('smartcash'),  # Main project
        ] + [logging.getLogger(name) for name in self._managed_loggers]
        
        self._original_levels.clear()
        
        try:
            for logger in loggers_to_suppress:
                self._original_levels[logger.name] = logger.level
                if test_mode:
                    logger.setLevel(logging.CRITICAL)
                else:
                    # In normal mode, only suppress INFO and below
                    logger.setLevel(logging.WARNING)
            
            yield
            
        finally:
            # Restore all original levels
            for logger in loggers_to_suppress:
                if logger.name in self._original_levels:
                    logger.setLevel(self._original_levels[logger.name])
            
            self._suppression_active = False
            self._original_levels.clear()
    
    def setup_ui_logging_handler(self, 
                                 module_name: str,
                                 log_message_func: Callable[[str, str], None]) -> None:
        """Setup UI logging handler for a specific module.
        
        Args:
            module_name: Name of the module (e.g., 'colab', 'dependency')
            log_message_func: Function to call for logging (message, level)
        """
        if not log_message_func or not callable(log_message_func):
            return
            
        class UILogHandler(logging.Handler):
            """Custom logging handler that routes logs to UI log_output."""
            
            def __init__(self, log_function: Callable[[str, str], None]):
                super().__init__()
                self.log_function = log_function
                
            def emit(self, record):
                """Emit a log record to the UI log_output."""
                try:
                    # Skip if suppression is active and this is INFO or below
                    if (UILoggingManager()._suppression_active and 
                        record.levelno <= logging.INFO):
                        return
                        
                    # Format the message
                    message = self.format(record)
                    
                    # Map logging levels to UI log levels
                    level_mapping = {
                        logging.DEBUG: 'DEBUG',
                        logging.INFO: 'INFO', 
                        logging.WARNING: 'WARNING',
                        logging.ERROR: 'ERROR',
                        logging.CRITICAL: 'CRITICAL'
                    }
                    
                    ui_level = level_mapping.get(record.levelno, 'INFO')
                    
                    # Send to UI log function
                    self.log_function(message, ui_level)
                    
                except Exception:
                    # Silently ignore errors in logging to prevent recursion
                    pass
        
        # Remove existing handler for this module
        self.remove_ui_logging_handler(module_name)
        
        # Create and configure the UI handler
        ui_handler = UILogHandler(log_message_func)
        ui_handler.setLevel(logging.INFO)
        
        # Create a clean formatter
        formatter = logging.Formatter('%(name)s: %(message)s')
        ui_handler.setFormatter(formatter)
        
        # Store handler reference
        self._ui_handlers[module_name] = ui_handler
        
        # Configure relevant loggers
        loggers_to_configure = [
            f'smartcash.ui.setup.{module_name}',
            f'smartcash.ui.{module_name}',
            'smartcash.ui.components',
            'smartcash.ui.core'
        ]
        
        for logger_name in loggers_to_configure:
            logger = logging.getLogger(logger_name)
            
            # Remove existing console handlers to prevent console output
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and hasattr(handler, 'stream'):
                    # Check if it's stdout/stderr to avoid removing file handlers
                    import sys
                    if handler.stream in (sys.stdout, sys.stderr):
                        logger.removeHandler(handler)
            
            # Add our UI handler
            logger.addHandler(ui_handler)
            logger.setLevel(logging.INFO)
            
            # Prevent propagation to avoid duplicate logs
            logger.propagate = False
    
    def remove_ui_logging_handler(self, module_name: str) -> None:
        """Remove UI logging handler for a specific module."""
        if module_name in self._ui_handlers:
            handler = self._ui_handlers[module_name]
            
            # Remove from all loggers
            for logger_name in logging.Logger.manager.loggerDict:
                if isinstance(logging.Logger.manager.loggerDict[logger_name], logging.Logger):
                    logger = logging.getLogger(logger_name)
                    if handler in logger.handlers:
                        logger.removeHandler(handler)
            
            del self._ui_handlers[module_name]
    
    def cleanup_all_handlers(self) -> None:
        """Clean up all UI logging handlers."""
        for module_name in list(self._ui_handlers.keys()):
            self.remove_ui_logging_handler(module_name)


# Global instance
_ui_logging_manager = UILoggingManager()

# Convenience functions for easy import
def suppress_ui_initialization_logs(test_mode: bool = False):
    """Context manager to suppress logs during UI initialization."""
    return _ui_logging_manager.suppress_initialization_logs(test_mode)

def setup_ui_logging(module_name: str, log_message_func: Callable[[str, str], None]):
    """Setup UI logging for a module."""
    _ui_logging_manager.setup_ui_logging_handler(module_name, log_message_func)

def cleanup_ui_logging(module_name: str = None):
    """Clean up UI logging handlers."""
    if module_name:
        _ui_logging_manager.remove_ui_logging_handler(module_name)
    else:
        _ui_logging_manager.cleanup_all_handlers()

def get_ui_logging_manager() -> UILoggingManager:
    """Get the global UI logging manager instance."""
    return _ui_logging_manager