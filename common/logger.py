"""
File: smartcash/common/logger.py
Deskripsi: Sistem logging terpusat dengan dukungan emoji, warna, dan callback
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union, Callable, List
from enum import Enum, auto

class LogLevel(Enum):
    """Level log dengan emoji."""
    DEBUG = auto()
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class SmartCashLogger:
    """
    Logger untuk SmartCash dengan fitur:
    - Emoji untuk level log
    - Format teks berwarna
    - Output ke file dan console
    - Support callback untuk UI
    """
    
    # Emoji untuk log level
    EMOJIS = {
        LogLevel.DEBUG: '🐞',
        LogLevel.INFO: 'ℹ️',
        LogLevel.SUCCESS: '✅',
        LogLevel.WARNING: '⚠️',
        LogLevel.ERROR: '❌',
        LogLevel.CRITICAL: '🔥'
    }
    
    # Colors (ANSI color codes)
    COLORS = {
        LogLevel.DEBUG: '\033[90m',  # Gray
        LogLevel.INFO: '\033[0m',    # Default
        LogLevel.SUCCESS: '\033[92m', # Green
        LogLevel.WARNING: '\033[93m', # Yellow
        LogLevel.ERROR: '\033[91m',   # Red
        LogLevel.CRITICAL: '\033[91;1m' # Bold Red
    }
    
    # Reset ANSI color
    RESET_COLOR = '\033[0m'
    
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
                level: LogLevel = LogLevel.INFO,
                log_file: Optional[str] = None,
                use_colors: bool = True,
                use_emojis: bool = True,
                log_dir: str = 'logs'):
        """
        Inisialisasi SmartCashLogger.
        
        Args:
            name: Nama logger
            level: Level minimum log
            log_file: Path file log (auto-generated jika None)
            use_colors: Flag untuk menggunakan warna
            use_emojis: Flag untuk menggunakan emoji
            log_dir: Direktori untuk file log
        """
        self.name = name
        self.level = level
        self.use_colors = use_colors
        self.use_emojis = use_emojis
        self._callbacks = []
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVEL_MAPPING[level])
        
        # Hapus handler yang sudah ada
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Tambahkan console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVEL_MAPPING[level])
        self.logger.addHandler(console_handler)
        
        # Tambahkan file handler jika diperlukan
        if log_file or log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Otomatis generate nama file jika tidak ada
            if not log_file:
                log_file = f"{self.log_dir}/{name}_{time.strftime('%Y%m%d')}.log"
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.LEVEL_MAPPING[level])
            
            # Format sederhana untuk file
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format pesan log dengan emoji dan warna."""
        formatted = ""
        
        # Tambahkan emoji jika diperlukan
        if self.use_emojis and level in self.EMOJIS:
            formatted += f"{self.EMOJIS[level]} "
            
        # Tambahkan pesan
        formatted += message
        
        # Tambahkan warna jika di console dan diizinkan
        if self.use_colors and level in self.COLORS:
            colored = f"{self.COLORS[level]}{formatted}{self.RESET_COLOR}"
            # Return versi berwarna untuk console, plain untuk file
            return colored, formatted
            
        # Tidak ada warna
        return formatted, formatted
    
    def log(self, level: LogLevel, message: str) -> None:
        """
        Log pesan dengan level tertentu.
        
        Args:
            level: Level log
            message: Pesan yang akan di-log
        """
        # Format pesan
        console_msg, file_msg = self._format_message(level, message)
        
        # Map ke level logging standar
        std_level = self.LEVEL_MAPPING[level]
        
        # Log via Python logger
        self.logger.log(std_level, file_msg)
        
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

# Fungsi helper untuk mendapatkan logger
def get_logger(name: str, 
              level: LogLevel = LogLevel.INFO, 
              log_file: Optional[str] = None,
              use_colors: bool = True,
              use_emojis: bool = True,
              log_dir: str = 'logs') -> SmartCashLogger:
    """
    Dapatkan instance SmartCashLogger.
    
    Args:
        name: Nama logger
        level: Level minimum log
        log_file: Path file log (auto-generated jika None)f
        use_colors: Flag untuk menggunakan warna
        use_emojis: Flag untuk menggunakan emoji
        log_dir: Direktori untuk file log
        
    Returns:
        Instance SmartCashLogger
    """
    return SmartCashLogger(name, level, log_file, use_colors, use_emojis, log_dir)