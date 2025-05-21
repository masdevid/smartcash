"""
File: smartcash/common/logger.py
Deskripsi: Sistem logging sederhana dengan dukungan callback untuk integrasi UI
"""

import logging
import sys
from enum import Enum, auto
from typing import Dict, Optional, Union, Callable, List

class LogLevel(Enum):
    """Level log."""
    DEBUG = auto()
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class SmartCashLogger:
    """
    Logger sederhana untuk SmartCash dengan dukungan callback untuk UI
    """
    
    # Mapping ke logging level standar
    LEVEL_MAPPING = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.SUCCESS: logging.INFO,  # Custom level, map ke INFO
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL
    }
    
    def __init__(self, 
                name: str, 
                level: LogLevel = LogLevel.INFO):
        """
        Inisialisasi SmartCashLogger.
        
        Args:
            name: Nama logger
            level: Level minimum log
        """
        self.name = name
        self.level = level
        self._callbacks = []
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVEL_MAPPING[level])
        
        # Jika tidak ada handler, tambahkan console handler default
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.LEVEL_MAPPING[level])
            formatter = logging.Formatter(
                '%(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log(self, level: LogLevel, message: str) -> None:
        """
        Log pesan dengan level tertentu.
        
        Args:
            level: Level log
            message: Pesan yang akan di-log
        """
        # Map ke level logging standar
        std_level = self.LEVEL_MAPPING[level]
        
        # Log via Python logger
        self.logger.log(std_level, message)
        
        # Panggil callbacks jika ada
        for callback in self._callbacks:
            callback(level, message)
    
    def add_callback(self, callback: Callable[[LogLevel, str], None]) -> None:
        """
        Tambahkan callback untuk event log.
        
        Args:
            callback: Fungsi yang dipanggil saat log
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """
        Hapus callback.
        
        Args:
            callback: Callback yang akan dihapus
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    # Convenience methods
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
    
    def error(self, message: str) -> None:
        """Log pesan error."""
        self.log(LogLevel.ERROR, message)
    
    def critical(self, message: str) -> None:
        """Log pesan critical."""
        self.log(LogLevel.CRITICAL, message)
    
    def progress(self, iterable=None, desc="Processing", **kwargs):
        """
        Buat progress bar dan log progress.
        
        Args:
            iterable: Iterable untuk diiterasi
            desc: Deskripsi progress
            **kwargs: Arguments tambahan untuk tqdm
            
        Returns:
            tqdm progress bar atau iterable asli jika tqdm tidak ada
        """
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, desc=desc, **kwargs)
        except ImportError:
            self.warning("tqdm tidak ditemukan, progress tracking tidak aktif")
            return iterable

    def set_level(self, level: LogLevel) -> None:
        """
        Atur level logging untuk logger dan semua handlers.
        
        Args:
            level: LogLevel untuk diatur
        """
        self.level = level
        std_level = self.LEVEL_MAPPING[level]
        self.logger.setLevel(std_level)
        for handler in self.logger.handlers:
            handler.setLevel(std_level)

# Fungsi helper untuk mendapatkan logger
def get_logger(name: Optional[str] = None, 
              level: LogLevel = LogLevel.INFO) -> SmartCashLogger:
    """
    Dapatkan instance SmartCashLogger.
    
    Args:
        name: Nama logger (default: __name__ dari modul pemanggil)
        level: Level minimum log
        
    Returns:
        Instance SmartCashLogger
    """
    import inspect
    
    # Jika name tidak disediakan, gunakan __name__ dari modul pemanggil
    if name is None:
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
            if frame is not None:
                name = frame.f_globals.get('__name__', 'smartcash')
    
    # Jika masih None, gunakan default
    if name is None:
        name = 'smartcash'
    
    return SmartCashLogger(
        name=name,
        level=level
    )