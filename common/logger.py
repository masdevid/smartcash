"""
File: smartcash/common/logger.py
Deskripsi: Sistem logging terpadu dengan format pesan konsisten dan dukungan callback UI
"""

import logging
import sys
from enum import Enum, auto
from typing import Dict, Optional, Union, Callable, List
from datetime import datetime

class LogLevel(Enum):
    """Level log dengan mapping konsisten."""
    DEBUG = auto()
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class MessageFormatter:
    """Formatter pesan yang konsisten dengan emoji dan timestamp."""
    
    # Emoji mapping yang konsisten
    EMOJI_MAP = {
        LogLevel.DEBUG: "🔍",
        LogLevel.INFO: "ℹ️",
        LogLevel.SUCCESS: "✅",
        LogLevel.WARNING: "⚠️",
        LogLevel.ERROR: "❌",
        LogLevel.CRITICAL: "🔥"
    }
    
    # Color mapping untuk UI
    COLOR_MAP = {
        LogLevel.DEBUG: "#6c757d",
        LogLevel.INFO: "#007bff", 
        LogLevel.SUCCESS: "#28a745",
        LogLevel.WARNING: "#ffc107",
        LogLevel.ERROR: "#dc3545",
        LogLevel.CRITICAL: "#dc3545"
    }
    
    @classmethod
    def format_message(cls, level: LogLevel, message: str, 
                      include_timestamp: bool = True, 
                      include_emoji: bool = True) -> str:
        """Format pesan dengan timestamp dan emoji."""
        formatted_parts = []
        
        if include_timestamp:
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_parts.append(f"[{timestamp}]")
        
        if include_emoji:
            emoji = cls.EMOJI_MAP.get(level, "📝")
            formatted_parts.append(emoji)
        
        formatted_parts.append(message)
        return " ".join(formatted_parts)
    
    @classmethod
    def format_html_message(cls, level: LogLevel, message: str, 
                           include_timestamp: bool = True) -> str:
        """Format pesan untuk HTML UI dengan styling."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        emoji = cls.EMOJI_MAP.get(level, "📝")
        color = cls.COLOR_MAP.get(level, "#212529")
        
        html_parts = []
        
        if include_timestamp:
            html_parts.append(f'<span style="color:#6c757d">[{timestamp}]</span>')
        
        html_parts.append(f'<span>{emoji}</span>')
        html_parts.append(f'<span style="color:{color}">{message}</span>')
        
        return f"""
        <div style="margin:2px 0;padding:3px;border-radius:3px;">
            {" ".join(html_parts)}
        </div>
        """

class SmartCashLogger:
    """Logger utama SmartCash dengan format pesan terpadu."""
    
    # Mapping ke logging level standar
    LEVEL_MAPPING = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.SUCCESS: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL
    }
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        """
        Inisialisasi SmartCashLogger dengan formatter terpadu.
        
        Args:
            name: Nama logger
            level: Level minimum log
        """
        self.name = name
        self.level = level
        self._callbacks = []
        self.formatter = MessageFormatter()
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVEL_MAPPING[level])
        
        # Hapus handler existing untuk mencegah duplikasi
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Setup console handler dengan formatter terpadu
        self._setup_console_handler()
    
    def _setup_console_handler(self) -> None:
        """Setup console handler dengan format konsisten."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVEL_MAPPING[self.level])
        
        # Custom formatter yang menggunakan MessageFormatter
        class UnifiedFormatter(logging.Formatter):
            def format(self, record):
                # Ambil level dari record
                level_mapping = {
                    logging.DEBUG: LogLevel.DEBUG,
                    logging.INFO: LogLevel.INFO,
                    logging.WARNING: LogLevel.WARNING,
                    logging.ERROR: LogLevel.ERROR,
                    logging.CRITICAL: LogLevel.CRITICAL
                }
                
                # Deteksi SUCCESS dari message prefix
                level = level_mapping.get(record.levelno, LogLevel.INFO)
                if record.levelno == logging.INFO and record.getMessage().startswith("SUCCESS:"):
                    level = LogLevel.SUCCESS
                    record.msg = record.msg[8:]  # Remove SUCCESS: prefix
                
                return MessageFormatter.format_message(
                    level, record.getMessage(),
                    include_timestamp=True,
                    include_emoji=True
                )
        
        console_handler.setFormatter(UnifiedFormatter())
        self.logger.addHandler(console_handler)
    
    def log(self, level: LogLevel, message: str, exc_info=None) -> None:
        """Log pesan dengan level dan format terpadu.
        
        Args:
            level: Level log
            message: Pesan yang akan dicatat
            exc_info: Optional exception info tuple (type, value, traceback)
        """
        # Log via Python logger dengan prefix untuk SUCCESS
        std_level = self.LEVEL_MAPPING[level]
        log_message = f"SUCCESS: {message}" if level == LogLevel.SUCCESS else message
        
        # Gunakan exc_info jika disediakan
        if exc_info is not None:
            self.logger.log(std_level, log_message, exc_info=exc_info)
        else:
            self.logger.log(std_level, log_message)
        
        # Panggil callbacks dengan format terpadu
        for callback in self._callbacks:
            try:
                callback(level, message, exc_info=exc_info)
            except Exception as e:
                # Error dalam callback tidak boleh mengganggu logging utama
                sys.stderr.write(f"Logger callback error: {str(e)}\n")
    
    def add_callback(self, callback: Callable[[LogLevel, str, Optional[tuple]], None]) -> None:
        """Tambahkan callback untuk event log.
        
        Args:
            callback: Fungsi callback dengan signature (level: LogLevel, message: str, exc_info: Optional[tuple]) -> None
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Hapus callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_level(self, level: LogLevel) -> None:
        """Atur level logging."""
        self.level = level
        std_level = self.LEVEL_MAPPING[level]
        self.logger.setLevel(std_level)
        for handler in self.logger.handlers:
            handler.setLevel(std_level)
    
    # Convenience methods dengan format terpadu
    def debug(self, message: str) -> None:
        """Log pesan debug."""
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """Log pesan info."""
        self.log(LogLevel.INFO, message)
    
    def success(self, message: str) -> None:
        """Log pesan sukses."""
        self.log(LogLevel.SUCCESS, message)
    
    def warning(self, message: str) -> None:
        """Log pesan warning."""
        self.log(LogLevel.WARNING, message)
    
    def error(self, message: str, exc_info=None) -> None:
        """Log pesan error.
        
        Args:
            message: Pesan error
            exc_info: Optional exception info tuple (type, value, traceback)
        """
        self.log(LogLevel.ERROR, message, exc_info=exc_info)
    
    def critical(self, message: str) -> None:
        """Log pesan critical."""
        self.log(LogLevel.CRITICAL, message)
    
    def progress(self, iterable=None, desc="Processing", **kwargs):
        """Buat progress bar dengan logging terintegrasi."""
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, desc=desc, **kwargs)
        except ImportError:
            self.warning("tqdm tidak ditemukan, progress tracking tidak aktif")
            return iterable

def get_logger(name: Optional[str] = None, 
              level: LogLevel = LogLevel.INFO) -> SmartCashLogger:
    """
    Factory function untuk mendapatkan logger dengan format terpadu.
    
    Args:
        name: Nama logger (auto-detect dari caller jika None)
        level: Level minimum log
        
    Returns:
        Instance SmartCashLogger dengan format terpadu
    """
    import inspect
    
    # Auto-detect nama dari caller jika tidak disediakan
    if name is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'smartcash')
    
    name = name or 'smartcash'
    
    return SmartCashLogger(name=name, level=level)

# Export key classes untuk kompatibilitas
__all__ = [
    'SmartCashLogger', 'LogLevel', 'MessageFormatter', 'get_logger'
]