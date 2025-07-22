"""
File: smartcash/ui/logger.py
Unified UI logging system for SmartCash UI components.

Features:
- Rich UI logging with colors and emojis
- Namespace support for better organization
- Thread-safe operations
- Buffering for early logs
- No file logging (UI only)
"""

import logging
import sys
import inspect
import threading
from typing import Dict, Any, Optional, List, Set, Deque, Union, Tuple
from collections import deque
import hashlib
import json
from enum import Enum

# Type aliases
LoggerType = logging.Logger

class LogLevel(Enum):
    """Standard log levels with UI-specific additions."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    SUCCESS = 25  # Custom level between INFO and WARNING

class UILogger:
    """
    Unified logging interface for SmartCash UI with support for:
    - Module-level logging with namespaces
    - UI integration with colors and emojis
    - Buffered logging for initialization phase
    - Thread-safe operations
    """
    
    # Emoji mappings for log levels
    EMOJI_MAP = {
        'debug': '🔍',
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'critical': '🚨',
    }
    
    # Color mappings for log levels
    COLOR_MAP = {
        'debug': '#6c757d',
        'info': '#007bff',
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'critical': '#dc3545',
    }
    
    def __init__(self, name: str, ui_components: Optional[Dict[str, Any]] = None, 
                 level: Union[str, int, LogLevel] = LogLevel.INFO):
        """
        Initialize the UI Logger.
        
        Args:
            name: Logger name (usually __name__ of the module)
            ui_components: Dictionary of UI components (must contain 'log_output')
            level: Logging level (default: INFO)
        """
        self.name = name
        self.ui_components = ui_components or {}
        self._level = self._normalize_level(level)
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._suppressed = False
        
        # Setup the underlying logger
        self.logger = logging.getLogger(f"ui.{name}")
        self.logger.setLevel(self._level)
        self.logger.propagate = False  # Prevent propagation to parent loggers
        
        # Add console handler if no handlers are present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _normalize_level(self, level: Union[str, int, LogLevel]) -> int:
        """Normalize log level to integer."""
        if isinstance(level, LogLevel):
            return level.value
        if isinstance(level, str):
            level = level.upper()
            if hasattr(logging, level):
                return getattr(logging, level)
        return int(level) if isinstance(level, (int, str) and str(level).isdigit()) else logging.INFO
    
    def _log_to_ui(self, message: str, level: str = 'info') -> None:
        """Log message to UI output."""
        if not self.ui_components or not message:
            return
            
        try:
            # Modern approach: use operation_container['log']
            operation_container = self.ui_components.get('operation_container')
            if operation_container and callable(operation_container.get('log')):
                operation_container['log'](message, level)
                return
                
            # Legacy fallback: log_output (deprecated)
            log_output = self.ui_components.get('log_output')
            if log_output:
                emoji = self.EMOJI_MAP.get(level.lower(), 'ℹ️')
                formatted = f"{emoji} {message}\n"
                
                if hasattr(log_output, 'append_stdout'):
                    log_output.append_stdout(formatted)
                elif hasattr(log_output, 'value'):
                    log_output.value += formatted
        except Exception:
            pass  # Silently fail UI logging
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log('info', message, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message."""
        self._log('success', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with optional exception info."""
        self._log('error', message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log('critical', message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log('error', message, exc_info=True, **kwargs)
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Internal logging implementation."""
        if self._suppressed:
            with self._lock:
                self._buffer.append({
                    'level': level,
                    'message': message,
                    'kwargs': kwargs
                })
            return
            
        # Log to standard logger
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message, **kwargs)
        
        # Log to UI
        self._log_to_ui(message, level)
    
    def flush_buffer(self) -> None:
        """Flush buffered logs to output."""
        with self._lock:
            while self._buffer:
                entry = self._buffer.popleft()
                self._log(entry['level'], entry['message'], **entry['kwargs'])
    
    def suppress(self) -> None:
        """Enable log suppression (buffer logs instead of outputting)."""
        self._suppressed = True
    
    def unsuppress(self, flush: bool = True) -> None:
        """Disable log suppression and optionally flush buffered logs."""
        self._suppressed = False
        if flush:
            self.flush_buffer()


def get_ui_logger(name: Optional[str] = None, 
                 ui_components: Optional[Dict[str, Any]] = None,
                 level: Union[str, int, LogLevel] = LogLevel.INFO) -> UILogger:
    """
    Get or create a UI logger instance.
    
    Args:
        name: Logger name (defaults to calling module's name)
        ui_components: Dictionary of UI components
        level: Logging level (default: INFO)
        
    Returns:
        Configured UILogger instance
    """
    if name is None:
        # Get the caller's module name
        frame = inspect.currentframe()
        try:
            module = inspect.getmodule(frame.f_back if frame else None)
            name = module.__name__ if module else 'root'
        finally:
            del frame  # Avoid reference cycles
    
    namespace_id = get_namespace_id(ui_components)
    border_color = get_namespace_color(namespace_id) if namespace_id else None
    
    return UILogger(name, ui_components, level)

# Namespace management functions
def get_namespace_id(ui_components: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Generate a namespace ID based on UI components.
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        A unique namespace ID or None if components are not provided
    """
    if not ui_components:
        return None
    
    # Create a stable string representation of the components
    components_str = json.dumps(
        {k: str(v) for k, v in sorted(ui_components.items())}, 
        sort_keys=True
    )
    
    # Generate a hash of the components
    return hashlib.md5(components_str.encode('utf-8')).hexdigest()

def get_namespace_color(namespace_id: Optional[str]) -> Optional[str]:
    """
    Get a consistent color for a namespace.
    
    Args:
        namespace_id: The namespace ID
        
    Returns:
        A CSS color string or None if no namespace ID is provided
    """
    if not namespace_id:
        return None
    
    # Simple hash-based color generation
    hue = int(namespace_id[:8], 16) % 360
    return f'hsl({hue}, 70%, 60%)'

# Create a default logger instance
default_logger = get_ui_logger('smartcash.ui')

# Backward compatibility
get_logger = get_ui_logger
get_module_logger = get_ui_logger
