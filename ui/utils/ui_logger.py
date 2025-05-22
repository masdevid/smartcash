"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Logger UI yang diperbaiki dengan filtering spam dan debug logging yang lebih baik
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Set
from IPython.display import display, HTML
from datetime import datetime

from smartcash.ui.utils.ui_logger_namespace import format_log_message

__all__ = ['UILogger', 'create_ui_logger', 'get_current_ui_logger', 'log_to_ui']

class UILogger:
    """Logger UI dengan filtering spam dan kontrol output yang lebih baik."""
    
    # Spam filter patterns
    _SPAM_PATTERNS = {
        'drive_detection', 'google drive', 'symlink config', 'setup config structure',
        'config symlink', 'direktori config', 'template config', 'handler', 'inisialisasi'
    }
    
    def __init__(self, ui_components: Dict[str, Any], name: str = "ui_logger", 
                 log_to_file: bool = False, log_dir: str = "logs", log_level: int = logging.INFO):
        self.ui_components = ui_components
        self.name = name
        self.log_level = log_level
        self._in_log_to_ui = False
        self._message_history: Set[str] = set()  # Prevent duplicate messages
        
        # Setup Python logger dengan minimal handlers
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add console handler hanya untuk CRITICAL errors
        console_handler = logging.StreamHandler(sys.__stderr__)
        console_handler.setLevel(logging.CRITICAL)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(console_handler)
        
        # File logging jika diminta
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            self.log_file_path = log_file
        else:
            self.log_file_path = None
    
    def _is_spam_message(self, message: str) -> bool:
        """Check apakah message adalah spam berdasarkan pattern."""
        msg_lower = message.lower()
        return any(pattern in msg_lower for pattern in self._SPAM_PATTERNS)
    
    def _should_skip_message(self, message: str, level: str) -> bool:
        """Tentukan apakah message harus di-skip."""
        if not message or not message.strip():
            return True
            
        # Skip duplicate messages
        msg_key = f"{level}:{message}"
        if msg_key in self._message_history:
            return True
        
        # Add to history (dengan limit untuk mencegah memory leak)
        if len(self._message_history) > 1000:
            self._message_history.clear()
        self._message_history.add(msg_key)
        
        # Skip spam messages kecuali error/warning
        if level not in ('error', 'warning', 'critical') and self._is_spam_message(message):
            return True
            
        return False
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log pesan ke UI dengan filtering dan formatting yang lebih baik."""
        if self._in_log_to_ui or self._should_skip_message(message, level):
            return
        
        self._in_log_to_ui = True
        
        try:
            # Format message dengan namespace
            formatted_message = format_log_message(self.ui_components, message)
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Emoji dan color mapping
            emoji_map = {
                "debug": "🔍", "info": "ℹ️", "success": "✅", 
                "warning": "⚠️", "error": "❌", "critical": "🔥"
            }
            
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
            except ImportError:
                color_map = {"info": "#007bff", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545"}
            
            emoji = emoji_map.get(level, "ℹ️")
            color = color_map.get(level, "#212529")
            
            # HTML formatting dengan styling yang konsisten
            html_content = f"""
            <div style="margin:2px 0;padding:3px;border-radius:3px;font-family:'Monaco','Menlo',monospace;font-size:12px;">
                <span style="color:#6c757d;">[{timestamp}]</span> 
                <span>{emoji}</span> 
                <span style="color:{color};">{formatted_message}</span>
            </div>
            """
            
            # Output ke log_output dengan prioritas
            if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                with self.ui_components['log_output']:
                    display(HTML(html_content))
            elif 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                with self.ui_components['status']:
                    display(HTML(html_content))
                    
        finally:
            self._in_log_to_ui = False
    
    def debug(self, message: str) -> None:
        """Log debug message - hanya tampilkan jika level DEBUG."""
        if not message or not message.strip(): return
        if self.log_level <= logging.DEBUG:
            self._log_to_ui(message, "debug")
    
    def info(self, message: str) -> None:
        """Log info message."""
        if not message or not message.strip(): return
        self._log_to_ui(message, "info")
    
    def success(self, message: str) -> None:
        """Log success message."""
        if not message or not message.strip(): return
        self._log_to_ui(message, "success")
    
    def warning(self, message: str) -> None:
        """Log warning message.""" 
        if not message or not message.strip(): return
        self.logger.warning(message)  # Still log to Python logger
        self._log_to_ui(message, "warning")
    
    def error(self, message: str) -> None:
        """Log error message."""
        if not message or not message.strip(): return
        self.logger.error(message)  # Still log to Python logger
        self._log_to_ui(message, "error")
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        if not message or not message.strip(): return
        self.logger.critical(message)  # Still log to Python logger  
        self._log_to_ui(message, "critical")

def create_ui_logger(ui_components: Dict[str, Any], name: str = "ui_logger",
                    log_to_file: bool = False, redirect_stdout: bool = False,
                    log_dir: str = "logs", log_level: int = logging.INFO) -> UILogger:
    """Buat UILogger dengan konfigurasi yang dioptimalkan."""
    logger = UILogger(ui_components, name, log_to_file, log_dir, log_level)
    
    # Redirect stdout jika diminta dan ada status widget
    if redirect_stdout and 'log_output' in ui_components:
        _setup_minimal_stdout_redirect(ui_components)
    
    # Integrasi dengan SmartCashLogger jika tersedia
    try:
        from smartcash.common.logger import get_logger
        sc_logger = get_logger(name)
        
        def ui_log_callback(level, message):
            if not message or not message.strip(): return
            level_map = {'SUCCESS': 'success', 'WARNING': 'warning', 'ERROR': 'error', 'DEBUG': 'debug'}
            logger._log_to_ui(message, level_map.get(level.name, 'info'))
        
        sc_logger.add_callback(ui_log_callback)
        ui_components['smartcash_logger'] = sc_logger
    except ImportError:
        pass
    
    ui_components['logger'] = logger
    _register_current_ui_logger(logger)
    
    return logger

def _setup_minimal_stdout_redirect(ui_components: Dict[str, Any]) -> None:
    """Setup minimal stdout redirect tanpa spam."""
    if 'custom_stdout' in ui_components:
        return  # Sudah ter-setup
    
    class MinimalStdoutInterceptor:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.terminal = sys.__stdout__
            self._in_write = False
            
        def write(self, message):
            if self._in_write or not message.strip():
                return
                
            self._in_write = True
            try:
                # Write ke terminal asli
                self.terminal.write(message)
                
                # Filter spam messages
                if any(spam in message.lower() for spam in ['config', 'symlink', 'setup', 'inisialisasi']):
                    return
                
                # Display ke UI hanya untuk message penting
                if 'log_output' in self.ui_components:
                    with self.ui_components['log_output']:
                        display(HTML(f"<div style='font-family:monospace;font-size:12px;color:#333;'>{message.strip()}</div>"))
            finally:
                self._in_write = False
                
        def flush(self):
            self.terminal.flush()
    
    # Setup interceptor
    original_stdout = sys.stdout
    ui_components['original_stdout'] = original_stdout
    interceptor = MinimalStdoutInterceptor(ui_components)
    sys.stdout = interceptor
    ui_components['custom_stdout'] = interceptor

# Singleton current logger
_current_ui_logger = None

def _register_current_ui_logger(logger: UILogger) -> None:
    global _current_ui_logger
    _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]:
    return _current_ui_logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """Log pesan ke UI secara langsung tanpa spam."""
    if not ui_components or not message or not message.strip():
        return
    
    # Skip spam messages
    spam_patterns = {'drive', 'config', 'symlink', 'setup', 'handler'}
    if level in ('info', 'debug') and any(pattern in message.lower() for pattern in spam_patterns):
        return
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    emoji_map = {"debug": "🔍", "info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    icon = icon or emoji_map.get(level, "ℹ️")
    
    formatted_message = format_log_message(ui_components, message)
    
    try:
        from smartcash.ui.utils.constants import COLORS
        color = {"info": "#007bff", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545"}.get(level, "#212529")
    except ImportError:
        color = "#212529"
    
    html_content = f"""
    <div style="margin:2px 0;padding:3px;font-family:monospace;font-size:12px;">
        <span style="color:#6c757d">[{timestamp}]</span> 
        <span>{icon}</span> 
        <span style="color:{color}">{formatted_message}</span>
    </div>
    """
    
    # Priority: log_output > status > print
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        with ui_components['log_output']:
            display(HTML(html_content))
    elif 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        with ui_components['status']:
            display(HTML(html_content))
    else:
        print(f"[{timestamp}] {icon} {formatted_message}")