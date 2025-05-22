"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: UI Logger dengan stdout interception yang lebih efektif untuk mencegah log muncul di console
"""

import logging
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Union
from IPython.display import display, HTML
import ipywidgets as widgets
from datetime import datetime
from smartcash.ui.utils.ui_logger_namespace import format_log_message

__all__ = [
    'UILogger', 
    'create_ui_logger', 
    'get_current_ui_logger',
    'log_to_ui',
    'intercept_stdout_to_ui',
    'restore_stdout'
]

class UILogger:
    """UI Logger dengan stdout interception yang diperbaiki"""
    
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
        
        # Setup Python logger dengan handler minimal
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Hanya tambahkan file handler jika diminta
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file_path = log_file
        else:
            self.log_file_path = None
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log ke UI tanpa stdout interference"""
        if not message or not message.strip() or self._in_log_to_ui:
            return
            
        self._in_log_to_ui = True
        
        try:
            formatted_message = format_log_message(self.ui_components, message)
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            emoji_map = {
                "debug": "🔍", "info": "ℹ️", "success": "✅",
                "warning": "⚠️", "error": "❌", "critical": "🔥"
            }
            emoji = emoji_map.get(level, "ℹ️")
            
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
            
            formatted_html = f"""
            <div style="margin:2px 0;padding:3px;border-radius:3px;">
                <span style="color:#6c757d">[{timestamp}]</span> 
                <span>{emoji}</span> 
                <span style="color:{color}">{formatted_message}</span>
            </div>
            """
            
            # Output prioritas: log_output -> status -> fallback
            if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                with self.ui_components['log_output']:
                    display(HTML(formatted_html))
            elif 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                with self.ui_components['status']:
                    display(HTML(formatted_html))
            
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
    """Create UI logger dengan stdout interception yang diperbaiki"""
    
    logger = UILogger(ui_components, name, log_to_file, log_dir, log_level)
    
    # Redirect stdout jika diminta
    if redirect_stdout:
        intercept_stdout_to_ui(ui_components)
    
    # Suppress root logger untuk mencegah duplicate output
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear semua handlers
    root_logger.setLevel(logging.CRITICAL)  # Set ke level tinggi
    
    ui_components['logger'] = logger
    _register_current_ui_logger(logger)
    
    return logger

def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """Intercept stdout yang lebih agresif"""
    
    if 'custom_stdout' in ui_components and ui_components.get('custom_stdout') == sys.stdout:
        return
    
    class AggressiveUIStdoutInterceptor:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.terminal = sys.__stdout__
            self.buffer = ""
            self._in_write = False
            
            # Filter patterns yang lebih komprehensif
            self.ignore_patterns = [
                'DEBUG:', '[DEBUG]', 'INFO:', '[INFO]', 'WARNING:', '[WARNING]',
                'Using TensorFlow', 'Colab notebook', 'Your session crashed',
                'Executing in eager mode', 'TensorFlow', 'NumExpr', 'Running on',
                '/usr/local/lib', 'Config file not found', 'inisialisasi',
                'setup', 'handler', 'initializing', 'Mounted at', 'Drive already',
                'terdaftar', 'terinisialisasi', 'berhasil', 'dibuat', 'disalin'
            ]
            
        def write(self, message):
            if self._in_write:
                return
                
            self._in_write = True
            
            try:
                # Filter aggressive untuk mencegah duplicate logs
                msg_strip = message.strip()
                if not msg_strip or len(msg_strip) < 3:
                    return
                
                # Skip semua pattern yang tidak diinginkan
                if any(pattern.lower() in msg_strip.lower() for pattern in self.ignore_patterns):
                    return
                
                # Skip emoji-based messages (biasanya dari UI logger)
                if any(emoji in msg_strip for emoji in ['✅', '❌', '⚠️', 'ℹ️', '🔍', '📁', '🔗', '📋']):
                    return
                
                self.buffer += message
                
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]
                    
                    for line in lines[:-1]:
                        if line.strip() and not any(pattern.lower() in line.lower() for pattern in self.ignore_patterns):
                            self._display_line(line)
                            
            finally:
                self._in_write = False
                
        def _display_line(self, line):
            try:
                formatted_line = format_log_message(self.ui_components, line)
                
                if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                    with self.ui_components['log_output']:
                        display(HTML(f"<div style='color:#212529'>{formatted_line}</div>"))
            except Exception:
                pass
        
        def flush(self):
            if self.buffer and self.buffer.strip():
                self._display_line(self.buffer)
                self.buffer = ""
        
        def isatty(self):
            return False
            
        def fileno(self):
            return self.terminal.fileno()
    
    # Replace stdout
    original_stdout = sys.stdout
    ui_components['original_stdout'] = original_stdout
    
    interceptor = AggressiveUIStdoutInterceptor(ui_components)
    sys.stdout = interceptor
    ui_components['custom_stdout'] = interceptor

def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """Restore stdout ke original"""
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

# Global logger reference
_current_ui_logger = None

def _register_current_ui_logger(logger: UILogger) -> None:
    global _current_ui_logger
    _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]:
    return _current_ui_logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """Direct log ke UI tanpa stdout interference"""
    if not ui_components or not message or not message.strip():
        return
        
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if icon is None:
        emoji_map = {"debug": "🔍", "info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "critical": "🔥"}
        icon = emoji_map.get(level, "ℹ️")
    
    formatted_message = format_log_message(ui_components, message)
    
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
    
    formatted_html = f"""
    <div style="margin:2px 0;padding:3px;border-radius:3px;">
        <span style="color:#6c757d">[{timestamp}]</span> 
        <span>{icon}</span> 
        <span style="color:{color}">{formatted_message}</span>
    </div>
    """
    
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        with ui_components['log_output']:
            display(HTML(formatted_html))
    elif 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        with ui_components['status']:
            display(HTML(formatted_html))
    elif 'output' in ui_components and hasattr(ui_components['output'], 'clear_output'):
        with ui_components['output']:
            display(HTML(formatted_html))