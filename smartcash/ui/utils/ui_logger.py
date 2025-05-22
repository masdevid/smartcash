"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: UI Logger terintegrasi dengan format pesan terpadu dari common logger
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from IPython.display import display, HTML
from datetime import datetime

# Import formatter terpadu dari common logger
from smartcash.common.logger import MessageFormatter, LogLevel, get_logger
from smartcash.ui.utils.ui_logger_namespace import format_log_message

class UILogger:
    """UI Logger dengan format pesan terpadu dan namespace support."""
    
    def __init__(self, ui_components: Dict[str, Any], name: str = "ui_logger",
                 log_to_file: bool = False, log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Inisialisasi UILogger dengan formatter terpadu.
        
        Args:
            ui_components: Dictionary komponen UI
            name: Nama logger
            log_to_file: Flag untuk logging ke file
            log_dir: Direktori log file
            log_level: Level logging
        """
        self.ui_components = ui_components
        self.name = name
        self.log_level = log_level
        self._in_log_to_ui = False
        self.formatter = MessageFormatter()
        
        # Setup logger dengan format terpadu
        self.common_logger = get_logger(name, self._map_to_common_level(log_level))
        
        # Setup file logging jika diperlukan
        if log_to_file:
            self._setup_file_logging(log_dir)
    
    def _map_to_common_level(self, std_level: int) -> LogLevel:
        """Map standard logging level ke LogLevel enum."""
        mapping = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARNING,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.CRITICAL
        }
        return mapping.get(std_level, LogLevel.INFO)
    
    def _setup_file_logging(self, log_dir: str) -> None:
        """Setup file logging handler."""
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            log_file = log_path / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            
            # Gunakan formatter standard untuk file
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            
            # Tambahkan ke common logger
            self.common_logger.logger.addHandler(file_handler)
            self.log_file_path = log_file
        except Exception as e:
            sys.stderr.write(f"Error setup file logging: {str(e)}\n")
            self.log_file_path = None
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log pesan ke UI dengan format terpadu."""
        if not message or not message.strip() or self._in_log_to_ui:
            return
        
        self._in_log_to_ui = True
        
        try:
            # Map level string ke LogLevel enum
            level_mapping = {
                "debug": LogLevel.DEBUG,
                "info": LogLevel.INFO,
                "success": LogLevel.SUCCESS,
                "warning": LogLevel.WARNING,
                "error": LogLevel.ERROR,
                "critical": LogLevel.CRITICAL
            }
            log_level = level_mapping.get(level, LogLevel.INFO)
            
            # Format pesan dengan namespace
            formatted_message = format_log_message(self.ui_components, message)
            
            # Gunakan MessageFormatter untuk HTML
            html_content = self.formatter.format_html_message(log_level, formatted_message)
            
            # Prioritas output: log_output > status > fallback
            if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                with self.ui_components['log_output']:
                    display(HTML(html_content))
            elif 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                with self.ui_components['status']:
                    self._display_status_message(log_level, formatted_message)
            else:
                # Fallback ke stdout dengan format console
                console_message = self.formatter.format_message(log_level, formatted_message)
                sys.__stdout__.write(f"{console_message}\n")
                sys.__stdout__.flush()
        
        finally:
            self._in_log_to_ui = False
    
    def _display_status_message(self, level: LogLevel, message: str) -> None:
        """Display message ke status widget dengan fallback."""
        try:
            from smartcash.ui.utils.alert_utils import create_status_indicator
            level_str = level.name.lower()
            emoji = self.formatter.EMOJI_MAP.get(level, "📝")
            display(create_status_indicator(level_str, message, emoji))
        except ImportError:
            # Fallback ke HTML sederhana
            html_content = self.formatter.format_html_message(level, message)
            display(HTML(html_content))
    
    # Unified logging methods dengan format konsisten
    def debug(self, message: str) -> None:
        """Log debug message dengan format terpadu."""
        if not message or not message.strip():
            return
        self.common_logger.debug(message)
        if self.log_level <= logging.DEBUG:
            self._log_to_ui(message, "debug")
    
    def info(self, message: str) -> None:
        """Log info message dengan format terpadu."""
        if not message or not message.strip():
            return
        self.common_logger.info(message)
        self._log_to_ui(message, "info")
    
    def success(self, message: str) -> None:
        """Log success message dengan format terpadu."""
        if not message or not message.strip():
            return
        self.common_logger.success(message)
        self._log_to_ui(message, "success")
    
    def warning(self, message: str) -> None:
        """Log warning message dengan format terpadu."""
        if not message or not message.strip():
            return
        self.common_logger.warning(message)
        self._log_to_ui(message, "warning")
    
    def error(self, message: str) -> None:
        """Log error message dengan format terpadu."""
        if not message or not message.strip():
            return
        self.common_logger.error(message)
        self._log_to_ui(message, "error")
    
    def critical(self, message: str) -> None:
        """Log critical message dengan format terpadu."""
        if not message or not message.strip():
            return
        self.common_logger.critical(message)
        self._log_to_ui(message, "critical")
    
    def progress(self, iterable=None, desc="Processing", **kwargs):
        """Progress bar dengan logging terintegrasi."""
        return self.common_logger.progress(iterable, desc, **kwargs)
    
    def set_level(self, level: int) -> None:
        """Set logging level untuk UI dan common logger."""
        self.log_level = level
        common_level = self._map_to_common_level(level)
        self.common_logger.set_level(common_level)
        
        level_names = {
            logging.DEBUG: "DEBUG", logging.INFO: "INFO",
            logging.WARNING: "WARNING", logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL"
        }
        level_name = level_names.get(level, str(level))
        self.info(f"Log level diubah ke {level_name}")

def create_ui_logger(ui_components: Dict[str, Any], name: str = "ui_logger",
                    log_to_file: bool = False, redirect_stdout: bool = True,
                    log_dir: str = "logs", log_level: int = logging.INFO) -> UILogger:
    """
    Factory function untuk membuat UILogger dengan format terpadu.
    
    Args:
        ui_components: Dictionary komponen UI
        name: Nama logger
        log_to_file: Flag untuk file logging
        redirect_stdout: Flag untuk redirect stdout
        log_dir: Direktori log file
        log_level: Level logging
        
    Returns:
        UILogger instance dengan format terpadu
    """
    # Buat UI logger
    logger = UILogger(ui_components, name, log_to_file, log_dir, log_level)
    
    # Setup stdout redirection jika diperlukan
    if redirect_stdout and 'status' in ui_components:
        intercept_stdout_to_ui(ui_components)
    
    # Reset logging handlers
    _reset_logging_handlers()
    
    # Setup callback bridge antara common logger dan UI
    _setup_logger_bridge(logger, ui_components)
    
    # Store logger reference
    ui_components['logger'] = logger
    if 'logger_namespace' not in ui_components:
        ui_components['logger_namespace'] = name
    
    # Register global reference
    _register_current_ui_logger(logger)
    
    return logger

def _setup_logger_bridge(ui_logger: UILogger, ui_components: Dict[str, Any]) -> None:
    """Setup bridge antara common logger dan UI logger."""
    def ui_callback(level: LogLevel, message: str):
        """Callback untuk meneruskan log dari common ke UI."""
        if not message or not message.strip():
            return
        
        level_map = {
            LogLevel.DEBUG: "debug", LogLevel.INFO: "info",
            LogLevel.SUCCESS: "success", LogLevel.WARNING: "warning",
            LogLevel.ERROR: "error", LogLevel.CRITICAL: "critical"
        }
        ui_level = level_map.get(level, "info")
        ui_logger._log_to_ui(message, ui_level)
    
    # Tambahkan callback ke common logger
    ui_logger.common_logger.add_callback(ui_callback)

def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """Intercept stdout dengan format terpadu."""
    if ('status' not in ui_components or 
        not hasattr(ui_components['status'], 'clear_output') or
        ui_components.get('custom_stdout') == sys.stdout):
        return
    
    class UnifiedStdoutInterceptor:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.terminal = sys.__stdout__
            self.buffer = ""
            self.formatter = MessageFormatter()
            self._in_write = False
            
            # Filter patterns yang konsisten
            self.ignore_patterns = [
                'DEBUG:', '[DEBUG]', 'INFO:', '[INFO]', 'WARNING:', '[WARNING]',
                'Using TensorFlow backend', 'Colab notebook', 'Your session crashed',
                'TensorFlow', 'NumExpr', 'Running on', '/usr/local/lib'
            ]
        
        def write(self, message):
            if self._in_write:
                return
            
            self._in_write = True
            
            try:
                self.terminal.write(message)
                
                msg_strip = message.strip()
                if (not msg_strip or len(msg_strip) < 2 or
                    any(pattern in msg_strip for pattern in self.ignore_patterns)):
                    return
                
                # Filter namespace messages
                from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, KNOWN_NAMESPACES
                current_ns = get_namespace_id(self.ui_components)
                if current_ns:
                    other_ns = [ns_id for ns_id in KNOWN_NAMESPACES.values() if ns_id != current_ns]
                    if any(f"[{ns_id}]" in msg_strip for ns_id in other_ns):
                        return
                
                self.buffer += message
                
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]
                    
                    for line in lines[:-1]:
                        if line.strip():
                            self._display_line(line)
            finally:
                self._in_write = False
        
        def _display_line(self, line):
            """Display line dengan format terpadu."""
            try:
                formatted_line = format_log_message(self.ui_components, line)
                
                if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                    with self.ui_components['log_output']:
                        html_content = self.formatter.format_html_message(LogLevel.INFO, formatted_line)
                        display(HTML(html_content))
                else:
                    with self.ui_components['status']:
                        try:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            display(create_status_indicator("info", formatted_line))
                        except ImportError:
                            html_content = self.formatter.format_html_message(LogLevel.INFO, formatted_line)
                            display(HTML(html_content))
            except Exception as e:
                self.terminal.write(f"[UI STDOUT ERROR: {str(e)}] {line}\n")
        
        def flush(self):
            self.terminal.flush()
            if self.buffer and self.buffer.strip():
                self._display_line(self.buffer)
                self.buffer = ""
        
        def isatty(self): return False
        def fileno(self): return self.terminal.fileno()
    
    # Setup interceptor
    original_stdout = sys.stdout
    ui_components['original_stdout'] = original_stdout
    
    interceptor = UnifiedStdoutInterceptor(ui_components)
    sys.stdout = interceptor
    ui_components['custom_stdout'] = interceptor

def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """Restore stdout ke kondisi original."""
    if 'original_stdout' in ui_components:
        custom_stdout = ui_components.get('custom_stdout')
        
        sys.stdout = ui_components['original_stdout']
        
        ui_components.pop('original_stdout', None)
        ui_components.pop('custom_stdout', None)
        
        if custom_stdout and hasattr(custom_stdout, 'flush'):
            try:
                custom_stdout.flush()
            except:
                pass

def _reset_logging_handlers():
    """Reset logging handlers dengan format terpadu."""
    root_logger = logging.getLogger()
    
    # Remove existing StreamHandlers
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.WARNING)
    
    # Add minimal handler
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.__stderr__)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """
    Standalone function untuk log ke UI dengan format terpadu.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan log
        level: Level log (info, success, warning, error)
        icon: Custom icon (opsional)
    """
    if not ui_components or not message or not message.strip():
        return
    
    # Map level ke LogLevel enum
    level_mapping = {
        "debug": LogLevel.DEBUG, "info": LogLevel.INFO,
        "success": LogLevel.SUCCESS, "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR, "critical": LogLevel.CRITICAL
    }
    log_level = level_mapping.get(level, LogLevel.INFO)
    
    # Override icon jika ada
    formatter = MessageFormatter()
    if icon:
        original_emoji = formatter.EMOJI_MAP.get(log_level)
        formatter.EMOJI_MAP[log_level] = icon
    
    # Format message dengan namespace
    formatted_message = format_log_message(ui_components, message)
    
    # Generate HTML dengan format terpadu
    html_content = formatter.format_html_message(log_level, formatted_message)
    
    # Restore original emoji
    if icon and 'original_emoji' in locals():
        formatter.EMOJI_MAP[log_level] = original_emoji
    
    # Display ke UI
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        with ui_components['log_output']:
            display(HTML(html_content))
    elif 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        with ui_components['status']:
            try:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator(level, formatted_message, icon or formatter.EMOJI_MAP.get(log_level)))
            except ImportError:
                display(HTML(html_content))
    else:
        # Fallback ke console
        console_message = formatter.format_message(log_level, formatted_message)
        print(console_message)

# Global UI logger reference
_current_ui_logger = None

def _register_current_ui_logger(logger: UILogger) -> None:
    """Register current UI logger untuk referensi global."""
    global _current_ui_logger
    _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]:
    """Get current active UI logger."""
    return _current_ui_logger

# Export untuk kompatibilitas
__all__ = [
    'UILogger', 'create_ui_logger', 'get_current_ui_logger', 
    'log_to_ui', 'intercept_stdout_to_ui', 'restore_stdout'
]