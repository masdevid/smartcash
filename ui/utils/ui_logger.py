"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Fixed UI Logger dengan clean message output tanpa duplicate formatting
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from IPython.display import display, HTML
from datetime import datetime
from smartcash.ui.utils.ui_logger_namespace import format_log_message, _clean_message

__all__ = [
    'UILogger', 
    'create_ui_logger', 
    'get_current_ui_logger',
    'log_to_ui',
    'intercept_stdout_to_ui',
    'restore_stdout'
]

class UILogger:
    """Fixed UI Logger dengan clean message output tanpa duplicate formatting."""
    
    def __init__(self, 
                ui_components: Dict[str, Any], 
                name: str = "ui_logger",
                log_to_file: bool = False,
                log_dir: str = "logs",
                log_level: int = logging.INFO):
        
        self.ui_components = ui_components
        self.name = name
        self.log_level = log_level
        self._in_log_to_ui = False
        
        # Setup Python logger dengan minimal handlers
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers to prevent leaks
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Suppress root logger untuk prevent backend logs
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.CRITICAL)
        
        # Setup file handler jika diminta
        if log_to_file:
            self._setup_file_handler(log_dir)
        
        # Setup aggressive stdout suppression
        self._setup_stdout_suppression()
    
    def _setup_file_handler(self, log_dir: str) -> None:
        """Setup file handler untuk logging ke file."""
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file_path = log_file
        except Exception:
            self.log_file_path = None
    
    def _setup_stdout_suppression(self) -> None:
        """Setup aggressive stdout suppression untuk prevent backend logs."""
        # Store original stdout
        if not hasattr(self.ui_components, 'original_stdout'):
            self.ui_components['original_stdout'] = sys.stdout
        
        # Create aggressive suppression wrapper
        class AggressiveStdoutSuppressor:
            def __init__(self, ui_components):
                self.ui_components = ui_components
                self.original = sys.__stdout__
                
                # Extended ignore patterns untuk backend services
                self.ignore_patterns = [
                    'DEBUG:', '[DEBUG]', 'INFO:', '[INFO]', 'WARNING:', '[WARNING]',
                    'Using TensorFlow', 'Colab notebook', 'Your session crashed',
                    'Executing in eager mode', 'TensorFlow', 'NumExpr', 'Running on',
                    '/usr/local/lib', 'Config file not found', 'inisialisasi',
                    'setup', 'handler', 'initializing', 'Mounted at', 'Drive already',
                    'terdaftar', 'terinisialisasi', 'berhasil', 'dibuat', 'disalin',
                    'progress', 'Progress', 'PROGRESS', 'downloading', 'extracting',
                    'metadata', 'validation', 'step', 'Stage', 'Processing',
                    # '✅', '❌', '⚠️', 'ℹ️', '🔍', '📁', '🔗', '📋', '🚀', '📊',
                    # '📥', '📦', '🎉', '💾', '🧹', '🔄', '📍', '💡', '🌐',
                    'requests.', 'urllib3.', 'http.client', 'connectionpool'
                ]
            
            def write(self, message):
                # Aggressive filtering - suppress everything by default
                return
            
            def flush(self):
                pass
            
            def isatty(self):
                return False
            
            def fileno(self):
                return self.original.fileno()
        
        # Replace stdout with aggressive suppressor
        suppressor = AggressiveStdoutSuppressor(self.ui_components)
        sys.stdout = suppressor
        self.ui_components['stdout_suppressor'] = suppressor
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log ke UI dengan clean formatting tanpa duplicate."""
        if not message or not message.strip() or self._in_log_to_ui:
            return
            
        self._in_log_to_ui = True
        
        try:
            # Clean message dari duplicate formatting
            clean_message = _clean_message(message)
            
            # Format hanya timestamp dan emoji
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            emoji_map = {
                "debug": "🔍", "info": "ℹ️", "success": "✅",
                "warning": "⚠️", "error": "❌", "critical": "🔥"
            }
            emoji = emoji_map.get(level, "ℹ️")
            
            # Color mapping
            try:
                from smartcash.ui.utils.constants import COLORS
                color_map = {
                    "debug": COLORS.get("muted", "#6c757d"),
                    "info": COLORS.get("primary", "#007bff"),
                    "success": COLORS.get("success", "#28a745"),
                    "warning": COLORS.get("warning", "#ffc107"),
                    "error": COLORS.get("danger", "#dc3545"),
                    "critical": COLORS.get("danger", "#dc3545")
                }
                color = color_map.get(level, COLORS.get("text", "#212529"))
            except ImportError:
                color = "#212529"
            
            # Get namespace color untuk border
            try:
                from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, get_namespace_color
                namespace_id = get_namespace_id(self.ui_components)
                border_color = get_namespace_color(namespace_id) if namespace_id else color
            except ImportError:
                border_color = color
            
            # Clean HTML formatting - hanya timestamp, emoji, dan message
            formatted_html = f"""
            <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
                       background-color:rgba(248,249,250,0.8);
                       border-left:3px solid {border_color};
                       font-family: 'Courier New', monospace;
                       font-size: 13px;">
                <span style="color:#6c757d;font-size:11px;">[{timestamp}]</span> 
                <span style="font-size:14px;">{emoji}</span> 
                <span style="color:{color};margin-left:4px;">{clean_message}</span>
            </div>
            """
            
            # Output dengan priority: log_output -> status -> fallback
            output_widget = None
            if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                output_widget = self.ui_components['log_output']
            elif 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                output_widget = self.ui_components['status']
            
            if output_widget:
                with output_widget:
                    display(HTML(formatted_html))
            
        except Exception:
            # Silent fail untuk prevent recursive errors
            pass
        finally:
            self._in_log_to_ui = False
    
    def debug(self, message: str) -> None:
        if message and message.strip() and self.log_level <= logging.DEBUG:
            self._log_to_ui(message, "debug")
    
    def info(self, message: str) -> None:
        if message and message.strip():
            self._log_to_ui(message, "info")
    
    def success(self, message: str) -> None:
        if message and message.strip():
            self._log_to_ui(message, "success")
    
    def warning(self, message: str) -> None:
        if message and message.strip():
            self._log_to_ui(message, "warning")
    
    def error(self, message: str) -> None:
        if message and message.strip():
            self._log_to_ui(message, "error")
    
    def critical(self, message: str) -> None:
        if message and message.strip():
            self._log_to_ui(message, "critical")

def create_ui_logger(ui_components: Dict[str, Any], 
                    name: str = "ui_logger",
                    log_to_file: bool = False,
                    redirect_stdout: bool = True,
                    log_dir: str = "logs",
                    log_level: int = logging.INFO) -> UILogger:
    """Create fixed UI logger dengan clean message output."""
    
    logger = UILogger(ui_components, name, log_to_file, log_dir, log_level)
    
    # Additional stdout suppression jika diminta
    if redirect_stdout:
        intercept_stdout_to_ui(ui_components)
    
    # Global suppression untuk prevent any leaks
    _suppress_all_backend_logging()
    
    ui_components['logger'] = logger
    _register_current_ui_logger(logger)
    
    return logger

def _suppress_all_backend_logging() -> None:
    """Suppress semua backend logging untuk prevent console leaks."""
    # Suppress requests library
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    
    # Suppress common libraries
    for lib_name in ['requests', 'urllib3', 'http.client', 'requests.packages.urllib3']:
        try:
            logging.getLogger(lib_name).setLevel(logging.CRITICAL)
            logging.getLogger(lib_name).propagate = False
        except Exception:
            pass

def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """Enhanced stdout interception untuk prevent backend logs."""
    # Additional layer of protection - sudah ada di UILogger.__init__
    pass

def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """Restore stdout ke original state."""
    if 'original_stdout' in ui_components:
        sys.stdout = ui_components['original_stdout']
        ui_components.pop('original_stdout', None)
        ui_components.pop('stdout_suppressor', None)

# Global logger reference
_current_ui_logger = None

def _register_current_ui_logger(logger: UILogger) -> None:
    global _current_ui_logger
    _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]:
    return _current_ui_logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """Fixed direct log ke UI dengan clean formatting."""
    if not ui_components or not message or not message.strip():
        return
        
    # Clean message dari duplicate formatting
    clean_message = _clean_message(message)
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if icon is None:
        emoji_map = {"debug": "🔍", "info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "critical": "🔥"}
        icon = emoji_map.get(level, "ℹ️")
    
    try:
        from smartcash.ui.utils.constants import COLORS
        color_map = {
            "debug": COLORS.get("muted", "#6c757d"), "info": COLORS.get("primary", "#007bff"),
            "success": COLORS.get("success", "#28a745"), "warning": COLORS.get("warning", "#ffc107"),
            "error": COLORS.get("danger", "#dc3545"), "critical": COLORS.get("danger", "#dc3545")
        }
        color = color_map.get(level, COLORS.get("text", "#212529"))
    except ImportError:
        color = "#212529"
    
    # Get namespace color untuk border
    try:
        from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, get_namespace_color
        namespace_id = get_namespace_id(ui_components)
        border_color = get_namespace_color(namespace_id) if namespace_id else color
    except ImportError:
        border_color = color
    
    formatted_html = f"""
    <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
               background-color:rgba(248,249,250,0.8);
               border-left:3px solid {border_color};
               font-family: 'Courier New', monospace;
               font-size: 13px;">
        <span style="color:#6c757d;font-size:11px;">[{timestamp}]</span> 
        <span style="font-size:14px;">{icon}</span> 
        <span style="color:{color};margin-left:4px;">{clean_message}</span>
    </div>
    """
    
    # Output dengan priority handling
    output_targets = ['log_output', 'status', 'output']
    for target in output_targets:
        if target in ui_components and hasattr(ui_components[target], 'clear_output'):
            with ui_components[target]:
                display(HTML(formatted_html))
            break