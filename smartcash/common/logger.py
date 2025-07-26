"""
File: smartcash/common/logger.py
Deskripsi: Core logging functionality for SmartCash (UI-agnostic)

Features:
- File and console logging with environment detection
- Thread-safe operations
- Simple and efficient
- No UI dependencies
- Automatic Colab environment detection and configuration
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

def detect_environment() -> str:
    """
    Detect the current execution environment.
    
    Returns:
        str: Environment type ('colab', 'jupyter', 'local', 'unknown')
    """
    try:
        # Check for Google Colab
        if 'google.colab' in sys.modules:
            return 'colab'
        
        # Check for Colab environment variables
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            return 'colab'
            
        # Check for Colab-specific paths
        if os.path.exists('/content') and os.path.exists('/usr/local/lib/python*/dist-packages/google/colab'):
            return 'colab'
            
        # Check for Jupyter notebook
        if 'IPython' in sys.modules:
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                        return 'jupyter'
            except ImportError:
                pass
        
        # Check for Jupyter environment variables
        if 'JUPYTER_SERVER_ROOT' in os.environ or 'JPY_SESSION_NAME' in os.environ:
            return 'jupyter'
            
        # Default to local environment
        return 'local'
        
    except Exception:
        return 'unknown'

def should_use_file_logging() -> bool:
    """
    Determine if file logging should be enabled based on environment.
    
    Returns:
        bool: True if file logging should be used
    """
    env = detect_environment()
    
    # Enable file logging for Colab and unknown environments
    if env in ['colab', 'unknown']:
        return True
    
    # Check for specific environment variables that suggest file logging
    if os.environ.get('SMARTCASH_LOG_TO_FILE', '').lower() in ['true', '1', 'yes']:
        return True
        
    return False

def should_disable_console_logging() -> bool:
    """
    Determine if console logging should be disabled based on environment.
    
    Returns:
        bool: True if console logging should be disabled
    """
    env = detect_environment()
    
    # Disable console logging in Colab when UI is active
    if env == 'colab' and SmartCashLogger._ui_mode_active:
        return True
    
    # Check for explicit disable flag
    if os.environ.get('SMARTCASH_DISABLE_CONSOLE_LOG', '').lower() in ['true', '1', 'yes']:
        return True
        
    return False

def get_log_directory() -> str:
    """
    Get the appropriate log directory based on environment.
    
    Returns:
        str: Path to log directory
    """
    env = detect_environment()
    
    if env == 'colab':
        # Use /content/logs for Colab
        return '/content/logs'
    elif env == 'jupyter':
        # Use ./logs for Jupyter
        return './logs'
    else:
        # Use ./logs for local environment
        return './logs'

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
                 log_to_console: Optional[bool] = None,
                 log_to_file: Optional[bool] = None,
                 log_file: Optional[str] = None,
                 log_dir: Optional[str] = None):
        """Initialize the logger with environment-aware configuration
        
        Args:
            name: Logger name (usually __name__)
            level: Logging level
            log_to_console: Whether to log to console (None for auto-detection)
            log_to_file: Whether to log to file (None for auto-detection)
            log_file: Custom log file path (optional)
            log_dir: Directory for log files (None for auto-detection)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Prevent duplicate log messages by disabling propagation to root logger
        self.logger.propagate = False
        
        # Auto-detect settings based on environment if not explicitly provided
        if log_to_console is None:
            log_to_console = not should_disable_console_logging()
        
        if log_to_file is None:
            log_to_file = should_use_file_logging()
            
        if log_dir is None:
            log_dir = get_log_directory()
        
        # Store environment info for reference
        self._environment = detect_environment()
        self._auto_configured = {
            'console': log_to_console,
            'file': log_to_file,
            'log_dir': log_dir
        }
        
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
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the detected environment and configuration.
        
        Returns:
            Dict containing environment info and auto-configured settings
        """
        return {
            'environment': self._environment,
            'auto_configured': self._auto_configured.copy(),
            'ui_mode_active': self._ui_mode_active,
            'handlers_count': len(self.logger.handlers),
            'handlers': [type(h).__name__ for h in self.logger.handlers]
        }

# Default logger instance
logger = SmartCashLogger('smartcash')

def get_logger(name: str = None, 
              level: Union[str, int, LogLevel] = None,
              log_to_console: Optional[bool] = None,
              log_to_file: Optional[bool] = None,
              log_file: Optional[str] = None,
              log_dir: Optional[str] = None) -> SmartCashLogger:
    """Get a logger instance with environment-aware configuration
    
    Args:
        name: Logger name (usually __name__). If None, returns the default logger.
        level: Logging level
        log_to_console: Whether to log to console (None for auto-detection)
        log_to_file: Whether to log to file (None for auto-detection)
        log_file: Custom log file path (optional)
        log_dir: Directory for log files (None for auto-detection)
        
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
                    log_to_console: Optional[bool] = None,
                    log_to_file: Optional[bool] = None,
                    log_file: Optional[str] = None,
                    log_dir: Optional[str] = None) -> SmartCashLogger:
    """Get a logger for a module with environment-aware configuration
    
    Args:
        module: Module name (usually __name__)
        level: Logging level
        log_to_console: Whether to log to console (None for auto-detection)
        log_to_file: Whether to log to file (None for auto-detection)
        log_file: Custom log file path (optional)
        log_dir: Directory for log files (None for auto-detection)
        
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

# Utility functions for environment-aware logging
def log_environment_info() -> None:
    """Log current environment information using the default logger."""
    env = detect_environment()
    use_file = should_use_file_logging()
    disable_console = should_disable_console_logging()
    log_dir = get_log_directory()
    
    logger.info(f"🌍 Environment detected: {env}")
    logger.info(f"📁 Log directory: {log_dir}")
    logger.info(f"📄 File logging: {'enabled' if use_file else 'disabled'}")
    logger.info(f"🖥️  Console logging: {'disabled' if disable_console else 'enabled'}")
    logger.info(f"🎛️  UI mode: {'active' if SmartCashLogger._ui_mode_active else 'inactive'}")

def configure_global_logging() -> None:
    """Configure global Python logging based on detected environment."""
    env = detect_environment()
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Configure based on environment
    if env == 'colab':
        # In Colab, use file logging and minimal console output
        log_dir = get_log_directory()
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'smartcash.log'), 
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
        ))
        root_logger.addHandler(file_handler)
        
        # Minimal console handler for critical errors only
        if not SmartCashLogger._ui_mode_active:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            root_logger.addHandler(console_handler)
    
    else:
        # For other environments, use normal console logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
        ))
        root_logger.addHandler(console_handler)