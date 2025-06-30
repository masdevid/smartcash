"""
File: smartcash/ui/utils/ui_logger.py
Consolidated logging system for SmartCash UI with namespace support.

Features:
- Module-level logger support with automatic namespace detection
- UI integration with colors and emojis
- File logging with rotation
- Stdout suppression and redirection
- Namespace management with colors
- Backward compatibility
"""

import logging
import sys
import os
import inspect
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Union, Callable, List, Set
from IPython.display import display, HTML
from datetime import datetime

# Type aliases
LoggerType = logging.Logger
T = TypeVar('T')

# Global state
_module_loggers = {}  # Cache for module loggers
_original_stdout = sys.stdout
_current_ui_logger = None

# Namespace management
class NamespaceManager:
    """Manage logging namespaces with colors and prefixes."""
    
    # Known namespaces with their prefixes and colors
    KNOWN_NAMESPACES = {
        # Setup & Environment
        "smartcash.ui.setup.env_config": "ENV",
        "smartcash.ui.setup.dependency": "DEPS",
        "smartcash.ui.setup": "SETUP",
        
        # Dataset related
        "smartcash.ui.dataset.downloader": "DOWNLOAD",
        "smartcash.ui.dataset.split": "SPLIT",
        "smartcash.ui.dataset.preprocessing": "PREPROC",
        "smartcash.ui.dataset.augmentation": "AUGMENT",
        "smartcash.ui.dataset.validation": "VALID",
        
        # Backend dataset modules
        "smartcash.dataset.preprocessing": "DATASET_PREPROC",
        "smartcash.dataset.augmentation": "DATASET_AUGMENT",
        "smartcash.dataset.validation": "DATASET_VALID",
        "smartcash.dataset.analysis": "DATASET_ANALYZE",
        "smartcash.dataset.downloader": "DATASET_DOWNLOADER",
        "smartcash.dataset.file_renamer": "DATASET_RENAME",
        "smartcash.dataset.organizer": "DATASET_ORG",
    }
    
    # Color scheme for namespaces
    NAMESPACE_COLORS = {
        # UI Components
        'SETUP': '#3498db',
        'ENV': '#2ecc71',
        'DEPS': '#9b59b6',
        'DOWNLOAD': '#e67e22',
        'PREPROC': '#1abc9c',
        'AUGMENT': '#f1c40f',
        'VALID': '#e74c3c',
        'SPLIT': '#e91e63',
        
        # Backend components
        'DATASET_PREPROC': '#16a085',
        'DATASET_AUGMENT': '#f39c12',
        'DATASET_VALID': '#c0392b',
        'DATASET_ANALYZE': '#8e44ad',
        'DATASET_DOWNLOADER': '#d35400',
        'DATASET_RENAME': '#27ae60',
        'DATASET_ORG': '#2980b9',
    }
    
    @classmethod
    def get_namespace_id(cls, module_name: str) -> str:
        """Get the namespace ID for a module."""
        # Try exact match first
        if module_name in cls.KNOWN_NAMESPACES:
            return cls.KNOWN_NAMESPACES[module_name]
        
        # Try partial match
        for ns, ns_id in cls.KNOWN_NAMESPACES.items():
            if module_name.startswith(ns):
                return ns_id
        
        # Default to the last part of the module name
        return module_name.split('.')[-1].upper() if module_name else 'UNKNOWN'
    
    @classmethod
    def get_namespace_color(cls, namespace_id: str) -> str:
        """Get the color for a namespace ID."""
        return cls.NAMESPACE_COLORS.get(namespace_id, '#95a5a6')
    
    @classmethod
    def register_namespace(cls, module_name: str, namespace_id: str, color: str = None) -> None:
        """Register a new namespace."""
        cls.KNOWN_NAMESPACES[module_name] = namespace_id
        if color:
            cls.NAMESPACE_COLORS[namespace_id] = color


# Log suppression utilities
class LogSuppressor:
    """Handle log suppression and redirection."""
    
    @staticmethod
    def setup_aggressive_log_suppression() -> None:
        """Setup aggressive log suppression for backend services."""
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.CRITICAL)
        root.propagate = False
        
        if not root.handlers:
            root.addHandler(logging.NullHandler())
        
        # Suppress common noisy loggers
        for name in ['matplotlib', 'PIL', 'tensorflow', 'torch', 'h5py', 'asyncio']:
            logging.getLogger(name).setLevel(logging.WARNING)
    
    @staticmethod
    def suppress_ml_logs() -> None:
        """Suppress machine learning library logs."""
        for name in ['tensorflow', 'tensorboard', 'torch', 'transformers', 'datasets']:
            logging.getLogger(name).setLevel(logging.WARNING)
    
    @staticmethod
    def suppress_viz_logs() -> None:
        """Suppress visualization library logs."""
        for name in ['matplotlib', 'PIL', 'Pillow', 'plotly', 'bokeh']:
            logging.getLogger(name).setLevel(logging.WARNING)

# Global state and utilities
_clean_message = lambda msg: msg.strip().replace('\n', ' ').replace('\r', '')[:500] if msg else ""
_get_color = lambda level, default="#212529": {"debug": "#6c757d", "info": "#007bff", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545", "critical": "#dc3545"}.get(level, default)
_get_emoji = lambda level: {"debug": "🔍", "info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "critical": "🔥"}.get(level, "ℹ️")

# Deprecation warning for old logger_bridge
class LoggerBridge:
    """Deprecated. Use UILogger instead."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LoggerBridge is deprecated. Use UILogger instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._logger = UILogger(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self._logger, name)

class UILogger:
    """
    Unified logging interface for SmartCash UI with support for:
    - Module-level logging with namespaces
    - UI integration with colors and emojis
    - File logging with rotation
    - Stdout suppression and redirection
    - Buffered logging for initialization phase
    """
    """Optimized UI Logger with full functionality.
    
    This logger provides a unified interface for both console and UI logging,
    with support for different log levels, colors, and emojis.
    """
    
    def __init__(self, ui_components: Dict[str, Any], name: str = "ui_logger", 
                 log_to_file: bool = False, log_dir: str = "logs", 
                 log_level: int = logging.INFO, enable_buffering: bool = False):
        """Initialize the UI Logger.
        
        Args:
            ui_components: Dictionary containing UI components
            name: Logger name (usually __name__ of the module)
            log_to_file: Whether to log to a file
            log_dir: Directory for log files
            log_level: Logging level (default: INFO)
            enable_buffering: If True, buffer logs until flush_buffered_logs() is called
        """
        self.ui_components = ui_components or {}
        self.name = name
        self.log_level = log_level
        self._in_log = False
        self._buffered_logs = [] if enable_buffering else None
        
        # Get module name for namespace
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        self.module_name = module.__name__ if module else 'root'
        self.namespace = NamespaceManager.get_namespace_id(self.module_name)
        self.namespace_color = NamespaceManager.get_namespace_color(self.namespace)
        
        # Configure the underlying logger
        self.logger = logging.getLogger(f"{self.namespace}.{name}")
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # Set up file handler if requested
        if log_to_file:
            self._setup_file_handler(log_dir)
        
        # Set as global logger if not already set
        global _current_ui_logger
        if _current_ui_logger is None:
            _current_ui_logger = self
    
    def _setup_file_handler(self, log_dir: str) -> None:
        """Set up file handler for logging to a file with rotation.
        
        Args:
            log_dir: Directory to store log files
        """
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Use RotatingFileHandler for log rotation
            from logging.handlers import RotatingFileHandler
            
            log_file = log_path / f"{self.name}.log"
            handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=5,
                encoding='utf-8'
            )
            
            handler.setLevel(self.log_level)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            
            self.logger.addHandler(handler)
            self.log_file_path = log_file
            
        except Exception as e:
            self.logger.error(f"Failed to set up file logging: {e}", exc_info=True)
            self.log_file_path = None
    
    def _setup_stdout_suppression(self) -> None:
        """Set up stdout suppression to prevent unwanted console output."""
        if not hasattr(sys, '_original_stdout'):
            sys._original_stdout = sys.stdout
        
        class StdoutSuppressor:
            def __init__(self):
                self.original = sys._original_stdout
            
            def write(self, msg):
                # Only suppress if not in debug mode
                if self.original and not self.original.closed:
                    level = logging.INFO
                    if 'error' in msg.lower() or 'exception' in msg.lower():
                        level = logging.ERROR
                    elif 'warning' in msg.lower():
                        level = logging.WARNING
                    
                    if level >= self.log_level:
                        self.original.write(msg)
                
            def flush(self):
                if self.original and not self.original.closed:
                    self.original.flush()
                
            def isatty(self):
                return hasattr(self.original, 'isatty') and self.original.isatty()
                
            def fileno(self):
                return self.original.fileno() if hasattr(self.original, 'fileno') else -1
        
        # Set up suppressor
        sys.stdout = StdoutSuppressor()
        self.ui_components['stdout_suppressor'] = sys.stdout
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log a message to the UI with the specified level.
        
        Args:
            message: The message to log
            level: Log level (debug, info, warning, error, critical, success)
        """
        if not message or not message.strip() or self._in_log: 
            return
            
        # If buffering is enabled, store the log and return
        if self._buffered_logs is not None:
            self._buffered_logs.append((level, message))
            return
            
        self._in_log = True
        console_fallback = False
        
        try:
            # Prepare message with namespace and timestamp
            timestamp = datetime.now().astimezone().strftime('%H:%M:%S %Z')
            clean_msg = _clean_message(message)
            emoji = _get_emoji(level)
            
            # Format message with namespace and color
            namespace_tag = f"[{self.namespace}]" if hasattr(self, 'namespace') else ""
            console_msg = f"[{timestamp}] {emoji} {namespace_tag} {clean_msg}"
            
            try:
                # Try to render to UI with color and formatting
                color = self.namespace_color if hasattr(self, 'namespace_color') else _get_color(level)
                border_color = color
                
                try: 
                    from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, get_namespace_color
                    namespace_id = get_namespace_id(self.ui_components)
                    border_color = get_namespace_color(namespace_id) if namespace_id else color
                except ImportError: 
                    pass
                
                # Check if this is a duplicate message
                is_duplicate = hasattr(self, '_last_message') and self._last_message == clean_msg
                self._last_message = clean_msg
                
                # Add border for duplicates
                border_style = '2px solid #e9ecef' if is_duplicate else 'none'
                
                # Add namespace badge if available
                namespace_badge = ''
                try:
                    # Get namespace ID from the existing call at the top of the function
                    if 'namespace_id' in locals() and namespace_id:
                        # Look up the namespace ID in KNOWN_NAMESPACES
                        from smartcash.ui.utils.ui_logger_namespace import KNOWN_NAMESPACES
                        namespace = KNOWN_NAMESPACES.get(namespace_id, f"NS:{namespace_id[:4]}")
                        namespace_badge = (
                            f'<span style="display:inline-block;padding:1px 4px;margin:1px 4px 0 0;align-self:flex-start;'
                            f'background-color:#f1f3f5;color:#5f3dc4;border-radius:2px;'
                            f'font-size:10px;font-weight:500;line-height:1.2;white-space:nowrap;">'
                            f'{namespace}</span>'
                        )
                except (ImportError, NameError):
                    pass
                
                # Build the HTML with row layout
                html_parts = [
                    f'<div style="margin:0 0 1px 0;padding:4px 8px;border-radius:2px;display:flex;align-items:flex-start;'
                    f'background-color:rgba(248,249,250,0.8);border-left:2px solid {border_color};'
                    f'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;'
                    f'font-size:12px;line-height:1.5;word-break:break-word;white-space:pre-wrap;'
                    f'overflow-wrap:break-word;border-right:{border_style};border-left:{border_style};">',
                    # Icon (left side)
                    f'<div style="margin-right:6px;flex-shrink:0;align-self:flex-start;padding-top:2px;">{emoji}</div>',
                    # Main content (middle)
                    f'<div style="flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;">',
                    f'<div style="color:{color};display:flex;align-items:center;width:100%;">',
                    f'<div style="flex:1;min-width:0;display:flex;align-items:center;gap:4px;margin-right:8px;">',
                    f'{namespace_badge if namespace_badge else ""}',
                    f'<span style="flex:1;min-width:0;white-space:pre-wrap;word-break:break-word;text-align:left;line-height:1.5;">{clean_msg}</span>',
                    '</div>',  # End of message container
                    # Timestamp (right side)
                    f'<div style="color:#6c757d;font-size:10px;font-family:monospace;white-space:nowrap;line-height:1.5;flex-shrink:0;">',
                    f'{timestamp}',
                    '</div>',  # End of timestamp
                    '</div>',  # End of message row
                    '</div>',  # End of main content
                    '</div>'  # End of log entry
                ]
                
                html = ''.join(html_parts)
                
                widget = next(
                    (self.ui_components[k] for k in ['log_output', 'status', 'output'] 
                     if k in self.ui_components and hasattr(self.ui_components[k], 'clear_output')), 
                    None
                )
                
                if widget: 
                    try:
                        with widget: 
                            display(HTML(html))
                        return  # Berhasil render ke UI, keluar dari fungsi
                    except Exception as e:
                        console_fallback = True
                        raise  # Lanjut ke blok except terluar
                else:
                    console_fallback = True
                        
            except Exception:
                console_fallback = True
                raise  # Lanjut ke blok except terluar
                
        except Exception:
            # Fallback ke console jika ada error
            if console_fallback or not any(k in self.ui_components for k in ['log_output', 'status', 'output']):
                try:
                    # Coba gunakan original_stdout jika ada
                    original_stdout = self.ui_components.get('original_stdout', sys.__stdout__)
                    print(console_msg, file=original_stdout)
                except:
                    # Fallback ke sys.__stderr__ jika masih gagal
                    print(f"[{level.upper()}] {message}", file=sys.__stderr__)
        finally: 
            self._in_log = False
    
    # One-liner logging methods
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message.
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(message, "debug")
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message.
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(message, "info")
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message.
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(message, "warning")
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with optional exception info.
        
        Args:
            message: The message to log
            exc_info: Whether to include exception info
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(message, "error")
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message.
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(message, "critical")
        self.logger.critical(message, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message (UI only).
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(f"✅ {message}", "info")
        self.logger.info(f"SUCCESS: {message}", **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback.
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log_to_ui(message, "error")
        self.logger.exception(message, **kwargs)
        
    def flush_buffered_logs(self) -> None:
        """Flush any buffered logs to the UI and disable buffering.
        
        This will process all buffered logs in the order they were received
        and then disable buffering for future logs.
        """
        if self._buffered_logs is not None:
            buffered = self._buffered_logs
            self._buffered_logs = None  # Disable buffering
            
            # Replay buffered logs
            for level, message in buffered:
                self._log_to_ui(message, level)
    
    def is_buffering(self) -> bool:
        """Check if the logger is currently buffering logs.
        
        Returns:
            bool: True if buffering is enabled, False otherwise
        """
        return self._buffered_logs is not None
    
    def clear_buffer(self) -> None:
        """Clear any buffered logs without processing them.
        
        This will remove all buffered logs but keep buffering enabled.
        """
        if self._buffered_logs is not None:
            self._buffered_logs.clear()
    
    def log(self, level: int, message: str, **kwargs) -> None:
        """Log a message with the specified integer level.
        
        Args:
            level: Integer level (e.g., logging.INFO, logging.ERROR)
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        level_name = logging.getLevelName(level).lower()

# Factory and utility functions
def create_ui_logger(ui_components: Dict[str, Any], name: str = "ui_logger", 
                   log_to_file: bool = False, redirect_stdout: bool = True, 
                   log_dir: str = "logs", log_level: int = logging.INFO) -> UILogger:
    """Create and configure a new UILogger instance.
        
    Args:
        ui_components: Dictionary of UI components
        name: Logger name (usually __name__ of the module)
        log_to_file: Whether to enable file logging
        redirect_stdout: Whether to redirect stdout to the logger
        log_dir: Directory for log files
        log_level: Minimum logging level
        
    Returns:
        Configured UILogger instance
    """
    return UILogger(
        ui_components=ui_components,
        name=name,
        log_to_file=log_to_file,
        log_dir=log_dir,
        log_level=log_level
    )

def get_logger(name: str = None, log_to_file: bool = False, 
               log_dir: str = "logs", log_level: int = logging.INFO) -> UILogger:
    """Get or create a logger with the given name.
        
    Args:
        name: Logger name (defaults to calling module's name)
        log_to_file: Whether to enable file logging
        log_dir: Directory for log files
        log_level: Minimum logging level
        
    Returns:
        Configured UILogger instance
    """
    if not name:
        # Get the caller's module name
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else 'root'
    
    if name in _module_loggers:
        return _module_loggers[name]
        
    # Create a logger with minimal UI components
    logger = UILogger(
        ui_components={},
        name=name,
        log_to_file=log_to_file,
        log_dir=log_dir,
        log_level=log_level
    )
    
    _module_loggers[name] = logger
    return logger

# Backward compatibility
get_module_logger = get_logger  # Alias for backward compatibility

# Setup global logging configuration
def setup_global_logging(ui_components: Dict[str, Any] = None, 
                       log_level: int = logging.INFO,
                       log_to_file: bool = False,
                       log_dir: str = "logs") -> None:
    """Setup global logging configuration.
        
    Args:
        ui_components: Dictionary of UI components (optional)
        log_level: Minimum logging level
        log_to_file: Whether to enable file logging
        log_dir: Directory for log files
    """
    # Basic logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Set up the global logger if UI components are provided
    if ui_components:
        global _current_ui_logger
        _current_ui_logger = UILogger(
            ui_components=ui_components,
            name="smartcash",
            log_to_file=log_to_file,
            log_dir=log_dir,
            log_level=log_level
        )
    
    # Suppress noisy loggers
    LogSuppressor.setup_aggressive_log_suppression()
    LogSuppressor.suppress_ml_logs()
    LogSuppressor.suppress_viz_logs()

# Clean up on module unload
import atexit
@atexit.register
def _cleanup():
    """Clean up resources when the module is unloaded."""
    global _current_ui_logger
    if _current_ui_logger:
        if hasattr(_current_ui_logger, 'log_file_path'):
            for handler in _current_ui_logger.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    _current_ui_logger.logger.removeHandler(handler)
        _current_ui_logger = None

def _suppress_all_backend_logging() -> None:
    """Suppress logging from common backend libraries."""
    for lib in ['requests', 'urllib3', 'http.client', 'requests.packages.urllib3']:
        logger = logging.getLogger(lib)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """Backward compatibility function. Use UILogger instead."""
    warnings.warn(
        "intercept_stdout_to_ui is deprecated. Use UILogger instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not _current_ui_logger:
        UILogger(ui_components)

def restore_stdout(ui_components: Dict[str, Any]) -> None: 
    sys.stdout = ui_components.pop('original_stdout', _original_stdout); ui_components.pop('stdout_suppressor', None)

def _register_current_ui_logger(logger: UILogger) -> None: 
    global _current_ui_logger; _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]: 
    return _current_ui_logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    if not ui_components or not message or not message.strip(): return
    clean_msg, timestamp, emoji, color = _clean_message(message), datetime.now().astimezone().strftime('%H:%M:%S %Z'), icon or _get_emoji(level), _get_color(level); border_color = color
    try: 
        from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, get_namespace_color; namespace_id = get_namespace_id(ui_components); border_color = get_namespace_color(namespace_id) if namespace_id else color
    except ImportError: pass
    
    html = f'<div style="margin:2px 0;padding:4px 8px;border-radius:4px;background-color:rgba(248,249,250,0.8);border-left:3px solid {border_color};font-family:\'Courier New\',monospace;font-size:13px;"><span style="color:#6c757d;font-size:11px;">[{timestamp}]</span> <span style="font-size:14px;">{emoji}</span> <span style="color:{color};margin-left:4px;">{clean_msg}</span></div>'
    
    widget = next((ui_components[k] for k in ['log_output', 'status', 'output'] if k in ui_components and hasattr(ui_components[k], 'clear_output')), None)
    if widget: 
        with widget: display(HTML(html))

# Module-level logger support
def get_module_logger(name: Optional[str] = None) -> 'UILogger':
    """Get or create a UILogger instance for the specified module.
    
    Args:
        name: Optional module name. If None, uses the caller's module name.
        
    Returns:
        Configured UILogger instance
    """
    if name is None:
        # Get the caller's module name
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else 'root'
    
    if name not in _module_loggers:
        # Create a new UILogger instance for this module
        ui_components = getattr(_current_ui_logger, 'ui_components', {}) if _current_ui_logger else {}
        _module_loggers[name] = UILogger(ui_components, name=name)
    
    return _module_loggers[name]


def setup_global_logging(ui_components: Dict[str, Any], **kwargs) -> 'UILogger':
    """Set up global logging with UI integration.
    
    Args:
        ui_components: Dictionary containing UI components
        **kwargs: Additional arguments to pass to UILogger constructor
        
    Returns:
        Configured UILogger instance
    """
    global _current_ui_logger
    _current_ui_logger = UILogger(ui_components, **kwargs)
    return _current_ui_logger


# Backward compatibility exports
__all__ = [
    'UILogger', 
    'create_ui_logger', 
    'get_current_ui_logger', 
    'log_to_ui', 
    'intercept_stdout_to_ui', 
    'restore_stdout',
    'get_module_logger',
    'setup_global_logging',
    'LoggerType'  # Export type alias
]