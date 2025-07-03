"""
File: smartcash/ui/core/shared/logger.py
Deskripsi: Enhanced UILogger dengan suppression support untuk prevent logs
sebelum log_output ready. Centralized logging dengan contextual emojis.
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from contextlib import contextmanager
import threading
from collections import deque

from smartcash.ui.utils.ui_logger import UILogger, get_module_logger


class EnhancedUILogger(UILogger):
    """Enhanced logger dengan auto-suppression dan buffering.
    
    Features:
    - 🔇 Auto-suppress logs sampai UI ready
    - 📦 Buffer logs saat suppressed
    - 🎯 Contextual emoji support
    - 🔄 Thread-safe operations
    - 📊 Log statistics tracking
    """
    
    # Emoji mappings untuk context
    CONTEXT_EMOJIS = {
        'init': '🚀',
        'setup': '🔧',
        'config': '📋',
        'save': '💾',
        'load': '📂',
        'reset': '🔄',
        'error': '❌',
        'warning': '⚠️',
        'success': '✅',
        'info': 'ℹ️',
        'debug': '🔍',
        'progress': '📊',
        'complete': '🎉',
        'cancel': '⏹️',
        'sync': '🔄',
        'network': '🌐',
        'file': '📄',
        'folder': '📁',
        'ui': '🎨',
        'handler': '🎯',
        'operation': '⚡',
    }
    
    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize enhanced logger."""
        super().__init__(name, level)
        
        # Suppression state
        self._suppressed = True  # Default suppressed until UI ready
        self._buffer = deque(maxlen=1000)  # Buffer untuk suppressed logs
        self._lock = threading.Lock()
        
        # Stats tracking
        self._stats = {
            'total': 0,
            'suppressed': 0,
            'buffered': 0,
            'by_level': {}
        }
        
        # Callbacks
        self._on_unsuppress_callbacks: List[Callable] = []
    
    # === Suppression Control ===
    
    def suppress(self) -> None:
        """Enable log suppression."""
        with self._lock:
            self._suppressed = True
            self.debug("🔇 Logging suppressed")
    
    def unsuppress(self, flush_buffer: bool = True) -> None:
        """Disable log suppression dan optionally flush buffer."""
        with self._lock:
            self._suppressed = False
            
            if flush_buffer and self._buffer:
                # Flush buffered logs
                buffer_copy = list(self._buffer)
                self._buffer.clear()
                
                for record in buffer_copy:
                    super().handle(record)
                
                self._stats['buffered'] = 0
            
            # Run callbacks
            for callback in self._on_unsuppress_callbacks:
                try:
                    callback()
                except Exception as e:
                    super().error(f"Callback error: {e}")
            
            self.debug(f"🔊 Logging enabled (flushed {len(buffer_copy) if flush_buffer else 0} logs)")
    
    @contextmanager
    def with_suppression(self, suppress: bool = True):
        """Context manager untuk temporary suppression."""
        original_state = self._suppressed
        
        if suppress:
            self.suppress()
        else:
            self.unsuppress(flush_buffer=False)
        
        try:
            yield
        finally:
            if original_state:
                self.suppress()
            else:
                self.unsuppress(flush_buffer=False)
    
    def on_unsuppress(self, callback: Callable) -> None:
        """Register callback untuk saat unsuppress."""
        self._on_unsuppress_callbacks.append(callback)
    
    # === Enhanced Logging Methods ===
    
    def handle(self, record: logging.LogRecord) -> None:
        """Handle log record dengan suppression check."""
        with self._lock:
            # Update stats
            self._stats['total'] += 1
            level_name = record.levelname
            self._stats['by_level'][level_name] = self._stats['by_level'].get(level_name, 0) + 1
            
            if self._suppressed:
                # Buffer the log
                self._buffer.append(record)
                self._stats['suppressed'] += 1
                self._stats['buffered'] = len(self._buffer)
            else:
                # Process normally
                super().handle(record)
    
    def log_with_context(self, 
                        level: int,
                        message: str,
                        context: Optional[str] = None,
                        **kwargs) -> None:
        """Log dengan contextual emoji."""
        # Get emoji untuk context
        emoji = ''
        if context:
            emoji = self.CONTEXT_EMOJIS.get(context.lower(), '')
        elif level == logging.ERROR:
            emoji = self.CONTEXT_EMOJIS['error']
        elif level == logging.WARNING:
            emoji = self.CONTEXT_EMOJIS['warning']
        elif level == logging.INFO:
            emoji = self.CONTEXT_EMOJIS['info']
        elif level == logging.DEBUG:
            emoji = self.CONTEXT_EMOJIS['debug']
        
        # Format message dengan emoji
        formatted_msg = f"{emoji} {message}" if emoji else message
        
        # Log it
        self.log(level, formatted_msg, **kwargs)
    
    # === Convenience Methods ===
    
    def info_context(self, message: str, context: str, **kwargs) -> None:
        """Info log dengan context."""
        self.log_with_context(logging.INFO, message, context, **kwargs)
    
    def error_context(self, message: str, context: str, **kwargs) -> None:
        """Error log dengan context."""
        self.log_with_context(logging.ERROR, message, context, **kwargs)
    
    def debug_context(self, message: str, context: str, **kwargs) -> None:
        """Debug log dengan context."""
        self.log_with_context(logging.DEBUG, message, context, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message."""
        self.log_with_context(logging.INFO, message, 'success', **kwargs)
    
    def progress(self, message: str, percentage: Optional[float] = None, **kwargs) -> None:
        """Log progress message."""
        if percentage is not None:
            message = f"{message} ({percentage:.1f}%)"
        self.log_with_context(logging.INFO, message, 'progress', **kwargs)
    
    # === Stats Methods ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset logging statistics."""
        with self._lock:
            self._stats = {
                'total': 0,
                'suppressed': 0,
                'buffered': 0,
                'by_level': {}
            }
    
    @property
    def is_suppressed(self) -> bool:
        """Check if currently suppressed."""
        return self._suppressed
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    def clear_buffer(self) -> None:
        """Clear log buffer without flushing."""
        with self._lock:
            self._buffer.clear()
            self._stats['buffered'] = 0


# === Global Logger Registry ===

_enhanced_loggers: Dict[str, EnhancedUILogger] = {}
_loggers_lock = threading.Lock()

def get_enhanced_logger(name: str, level: int = logging.INFO) -> EnhancedUILogger:
    """Get atau create enhanced logger instance.
    
    Args:
        name: Logger name (biasanya module path)
        level: Log level
        
    Returns:
        Enhanced logger instance
    """
    with _loggers_lock:
        if name not in _enhanced_loggers:
            _enhanced_loggers[name] = EnhancedUILogger(name, level)
        
        return _enhanced_loggers[name]

def suppress_all_loggers() -> None:
    """Suppress semua enhanced loggers."""
    with _loggers_lock:
        for logger in _enhanced_loggers.values():
            logger.suppress()

def unsuppress_all_loggers(flush_buffer: bool = True) -> None:
    """Unsuppress semua enhanced loggers."""
    with _loggers_lock:
        for logger in _enhanced_loggers.values():
            logger.unsuppress(flush_buffer)

def get_all_logger_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics dari semua loggers."""
    with _loggers_lock:
        return {
            name: logger.get_stats() 
            for name, logger in _enhanced_loggers.items()
        }