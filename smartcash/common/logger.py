"""
File: smartcash/common/logger.py
Deskripsi: Core logging functionality for SmartCash (UI-agnostic)

Features:
- File and console logging
- Thread-safe operations
- Simple and efficient
- No UI dependencies
"""

import logging
import sys
import os
from typing import Optional, Union, Dict, Any
from pathlib import Path
from enum import Enum

class LogLevel(Enum):
    """Standard log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # Aliases for backward compatibility
    WARN = logging.WARNING
    FATAL = logging.CRITICAL

class SmartCashLogger:
    """Core logger for SmartCash applications"""
    
    # Global flag to disable console logging when UI is active
    _ui_mode_active = False
    _ui_handler = None
    
    @classmethod
    def set_ui_mode(cls, active: bool, ui_handler=None):
        """Set UI mode to control console logging globally."""
        cls._ui_mode_active = active
        cls._ui_handler = ui_handler
    
    def __init__(self, name: str, 
                 level: Union[str, int, LogLevel] = LogLevel.INFO,
                 log_to_console: bool = True,
                 log_to_file: bool = False,
                 log_file: Optional[str] = None,
                 log_dir: str = 'logs'):
        """Initialize the logger
        
        Args:
            name: Logger name (usually __name__)
            level: Logging level
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_file: Custom log file path (optional)
            log_dir: Directory for log files (if log_file not specified)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger(level, log_to_console, log_to_file, log_file, log_dir)
    
    def _setup_logger(self, 
                     level: Union[str, int, LogLevel],
                     log_to_console: bool,
                     log_to_file: bool,
                     log_file: Optional[str],
                     log_dir: str) -> None:
        """Configure the logger with handlers"""
        # Clear existing handlers
        self.logger.handlers = []
        
        # Set log level
        safe_level = self._normalize_level(level)
        self.logger.setLevel(safe_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler (but check UI mode first)
        if log_to_console and not self._ui_mode_active:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(safe_level)
            self.logger.addHandler(console_handler)
        elif self._ui_mode_active and self._ui_handler:
            # Add UI handler instead of console when in UI mode
            self.logger.addHandler(self._ui_handler)
        
        # Add file handler
        if log_to_file:
            if not log_file:
                # Create log directory if it doesn't exist
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{self.name}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(safe_level)
            self.logger.addHandler(file_handler)
    
    def _normalize_level(self, level: Union[str, int, LogLevel]) -> int:
        """Normalize log level to integer"""
        if isinstance(level, LogLevel):
            return level.value
        if isinstance(level, int):
            return level if logging.NOTSET <= level <= logging.CRITICAL else logging.INFO
        if isinstance(level, str):
            level_upper = level.upper()
            if hasattr(logging, level_upper):
                return getattr(logging, level_upper)
        return logging.INFO
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message as INFO level with success styling"""
        self.logger.info(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        if 'exc_info' not in kwargs:
            kwargs['exc_info'] = True
        self.logger.error(message, **kwargs)
    
    def set_level(self, level: Union[str, int, LogLevel]):
        """Set log level for all handlers"""
        safe_level = self._normalize_level(level)
        self.logger.setLevel(safe_level)
        for handler in self.logger.handlers:
            handler.setLevel(safe_level)

# Default logger instance
logger = SmartCashLogger('smartcash')

def get_logger(name: str = None, 
              level: Union[str, int, LogLevel] = None,
              log_to_console: bool = True,
              log_to_file: bool = False,
              log_file: Optional[str] = None,
              log_dir: str = 'logs') -> SmartCashLogger:
    """Get a logger instance with the given configuration
    
    Args:
        name: Logger name (usually __name__). If None, returns the default logger.
        level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file: Custom log file path (optional)
        log_dir: Directory for log files (if log_file not specified)
        
    Returns:
        Configured SmartCashLogger instance
    """
    if name is None:
        return logger
    return SmartCashLogger(
        name=name,
        level=level or LogLevel.INFO,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_file=log_file,
        log_dir=log_dir
    )

def get_module_logger(module: str, 
                    level: Union[str, int, LogLevel] = None,
                    log_to_console: bool = True,
                    log_to_file: bool = False,
                    log_file: Optional[str] = None,
                    log_dir: str = 'logs') -> SmartCashLogger:
    """Get a logger for a module with the given configuration
    
    Args:
        module: Module name (usually __name__)
        level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file: Custom log file path (optional)
        log_dir: Directory for log files (if log_file not specified)
        
    Returns:
        Configured SmartCashLogger instance for the module
    """
    return get_logger(
        name=module,
        level=level,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_file=log_file,
        log_dir=log_dir
    )

def safe_log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info') -> bool:
    """Utility function for safe UI logging using modern operation_container.
    
    Args:
        ui_components: Dictionary containing either:
                      - 'operation_container' with 'log' method (modern approach)
                      - 'log_output' (legacy, deprecated)
        message: Message to log
        level: Log level (info, warning, error, etc.)
        
    Returns:
        bool: True if logging succeeded, False otherwise
    """
    try:
        # Modern approach: use operation_container['log']
        operation_container = ui_components.get('operation_container')
        if operation_container and callable(operation_container.get('log')):
            operation_container['log'](message, level)
            return True
            
        # Legacy fallback: log_output (deprecated)
        log_output = ui_components.get('log_output')
        if log_output:
            emoji_map = {
                'info': 'ℹ️', 'warning': '⚠️', 'error': '❌',
                'critical': '🚨', 'success': '✅', 'debug': '🔍'
            }
            
            emoji = emoji_map.get(level.lower(), 'ℹ️')
            formatted_message = f"{emoji} {message}\n"
            
            if hasattr(log_output, 'append_stdout'):
                log_output.append_stdout(formatted_message)
                return True
            elif hasattr(log_output, 'value'):
                log_output.value += formatted_message
                return True
                
        return False
        
    except Exception:
        return False

def create_ui_logger(name: str, ui_components: Dict[str, Any], level: Union[str, int, LogLevel] = LogLevel.INFO) -> SmartCashLogger:
    """Create logger dengan UI components integration"""
    logger = SmartCashLogger(name, level)
    logger.set_ui_components(ui_components)
    return logger