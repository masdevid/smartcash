"""
File: smartcash/ui/utils/ui_logger.py
Optimized one-liner UI Logger with backward compatibility
"""

import logging, sys, os; from pathlib import Path; from typing import Dict, Any, Optional; from IPython.display import display, HTML; from datetime import datetime

# Global state and utilities
_current_ui_logger, _original_stdout = None, sys.stdout
_clean_message = lambda msg: msg.strip().replace('\n', ' ').replace('\r', '')[:500] if msg else ""
_get_color = lambda level, default="#212529": {"debug": "#6c757d", "info": "#007bff", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545", "critical": "#dc3545"}.get(level, default)
_get_emoji = lambda level: {"debug": "🔍", "info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "critical": "🔥"}.get(level, "ℹ️")

class UILogger:
    """Optimized UI Logger - one-liner methods with full functionality"""
    
    def __init__(self, ui_components: Dict[str, Any], name: str = "ui_logger", log_to_file: bool = False, log_dir: str = "logs", log_level: int = logging.INFO):
        self.ui_components, self.name, self.log_level, self._in_log = ui_components, name, log_level, False; self.logger = logging.getLogger(name); [self.logger.removeHandler(h) for h in self.logger.handlers[:]]; self.logger.setLevel(log_level); logging.getLogger().handlers.clear(); logging.getLogger().setLevel(logging.CRITICAL); self._setup_file_handler(log_dir) if log_to_file else None; self._setup_stdout_suppression()
    
    def _setup_file_handler(self, log_dir: str) -> None:
        try: log_path = Path(log_dir); log_path.mkdir(parents=True, exist_ok=True); log_file = log_path / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"; handler = logging.FileHandler(log_file, encoding='utf-8'); handler.setLevel(self.log_level); handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')); self.logger.addHandler(handler); setattr(self, 'log_file_path', log_file)
        except Exception: setattr(self, 'log_file_path', None)
    
    def _setup_stdout_suppression(self) -> None:
        class StdoutSuppressor:
            def __init__(self): self.original = sys.__stdout__
            def write(self, msg): return  # Suppress all output
            def flush(self): pass
            def isatty(self): return False
            def fileno(self): return self.original.fileno()
        
        self.ui_components.setdefault('original_stdout', sys.stdout); sys.stdout = StdoutSuppressor(); self.ui_components['stdout_suppressor'] = sys.stdout
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        if not message or not message.strip() or self._in_log: return
        self._in_log = True
        try:
            clean_msg = _clean_message(message); timestamp = datetime.now().strftime('%H:%M:%S'); emoji = _get_emoji(level); color = _get_color(level); border_color = color
            try: 
                from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, get_namespace_color; namespace_id = get_namespace_id(self.ui_components); border_color = get_namespace_color(namespace_id) if namespace_id else color
            except ImportError: pass
            
            html = f'<div style="margin:2px 0;padding:4px 8px;border-radius:4px;background-color:rgba(248,249,250,0.8);border-left:3px solid {border_color};font-family:\'Courier New\',monospace;font-size:13px;"><span style="color:#6c757d;font-size:11px;">[{timestamp}]</span> <span style="font-size:14px;">{emoji}</span> <span style="color:{color};margin-left:4px;">{clean_msg}</span></div>'
            
            widget = next((self.ui_components[k] for k in ['log_output', 'status', 'output'] if k in self.ui_components and hasattr(self.ui_components[k], 'clear_output')), None)
            if widget: 
                with widget: display(HTML(html))
        except Exception: pass
        finally: self._in_log = False
    
    # One-liner logging methods
    def debug(self, message: str) -> None: self._log_to_ui(message, "debug") if message and message.strip() and self.log_level <= logging.DEBUG else None
    def info(self, message: str) -> None: self._log_to_ui(message, "info") if message and message.strip() else None
    def success(self, message: str) -> None: self._log_to_ui(message, "success") if message and message.strip() else None
    def warning(self, message: str) -> None: self._log_to_ui(message, "warning") if message and message.strip() else None
    def error(self, message: str) -> None: self._log_to_ui(message, "error") if message and message.strip() else None
    def critical(self, message: str) -> None: self._log_to_ui(message, "critical") if message and message.strip() else None

# One-liner factory and utility functions
def create_ui_logger(ui_components: Dict[str, Any], name: str = "ui_logger", log_to_file: bool = False, redirect_stdout: bool = True, log_dir: str = "logs", log_level: int = logging.INFO) -> UILogger:
    logger = UILogger(ui_components, name, log_to_file, log_dir, log_level); intercept_stdout_to_ui(ui_components) if redirect_stdout else None; _suppress_all_backend_logging(); ui_components.update({'logger': logger}); _register_current_ui_logger(logger); return logger

def _suppress_all_backend_logging() -> None:
    [setattr(logging.getLogger(lib), 'level', logging.CRITICAL) or setattr(logging.getLogger(lib), 'propagate', False) for lib in ['requests', 'urllib3', 'http.client', 'requests.packages.urllib3']]

def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None: pass  # Already handled in UILogger.__init__

def restore_stdout(ui_components: Dict[str, Any]) -> None: 
    sys.stdout = ui_components.pop('original_stdout', _original_stdout); ui_components.pop('stdout_suppressor', None)

def _register_current_ui_logger(logger: UILogger) -> None: 
    global _current_ui_logger; _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]: 
    return _current_ui_logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    if not ui_components or not message or not message.strip(): return
    clean_msg, timestamp, emoji, color = _clean_message(message), datetime.now().strftime('%H:%M:%S'), icon or _get_emoji(level), _get_color(level); border_color = color
    try: 
        from smartcash.ui.utils.ui_logger_namespace import get_namespace_id, get_namespace_color; namespace_id = get_namespace_id(ui_components); border_color = get_namespace_color(namespace_id) if namespace_id else color
    except ImportError: pass
    
    html = f'<div style="margin:2px 0;padding:4px 8px;border-radius:4px;background-color:rgba(248,249,250,0.8);border-left:3px solid {border_color};font-family:\'Courier New\',monospace;font-size:13px;"><span style="color:#6c757d;font-size:11px;">[{timestamp}]</span> <span style="font-size:14px;">{emoji}</span> <span style="color:{color};margin-left:4px;">{clean_msg}</span></div>'
    
    widget = next((ui_components[k] for k in ['log_output', 'status', 'output'] if k in ui_components and hasattr(ui_components[k], 'clear_output')), None)
    if widget: 
        with widget: display(HTML(html))

# Backward compatibility exports
__all__ = ['UILogger', 'create_ui_logger', 'get_current_ui_logger', 'log_to_ui', 'intercept_stdout_to_ui', 'restore_stdout']