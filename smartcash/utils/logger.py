# File: smartcash/utils/logger.py
# Author: Alfrida Sabar
# Deskripsi: Custom logger dengan emoji dan warna yang mewarisi logging.Logger
# dengan perbaikan untuk menampilkan log di file tanpa menampilkan di konsol

import logging
import sys
import os
from typing import Optional
from termcolor import colored
from datetime import datetime
from pathlib import Path

class SmartCashLogger(logging.Logger):
    """Custom logger untuk SmartCash project dengan emoji dan colored output."""
    
    EMOJIS = {
        'start': '🚀',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'data': '📊',
        'model': '🤖',
        'time': '⏱️',
        'metric': '📈',
        'save': '💾',
        'load': '📂'
    }
    
    def __init__(
        self, 
        name: str,
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_dir: str = "logs"
    ):
        """
        Inisialisasi logger.
        
        Args:
            name: Nama logger
            level: Level logging (default: INFO)
            log_to_file: Flag untuk mengaktifkan logging ke file
            log_to_console: Flag untuk mengaktifkan logging ke konsol
            log_dir: Direktori untuk menyimpan file log
        """
        # Inisialisasi logger dasar
        super().__init__(name, level)
        
        # Buat direktori log jika belum ada
        self.log_dir = Path(log_dir)
        if log_to_file:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup console handler jika diperlukan
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # Setup formatter
            console_formatter = logging.Formatter(
                '%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            
            # Tambahkan handler
            self.addHandler(console_handler)
            
        # Setup file handler jika diperlukan
        if log_to_file:
            # Buat nama file dengan timestamp
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"smartcash_{today}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            
            # Setup formatter dengan lebih detail untuk file
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            
            # Tambahkan handler
            self.addHandler(file_handler)
    
    def _format_message(
        self, 
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False
    ) -> str:
        """Format pesan dengan emoji dan warna."""
        emoji = self.EMOJIS.get(emoji_key, '')
        
        if highlight_values and any(char.isdigit() for char in msg):
            # Highlight angka dengan warna
            words = msg.split()
            for i, word in enumerate(words):
                if any(char.isdigit() for char in word):
                    words[i] = colored(word, 'yellow')
            msg = ' '.join(words)
        
        formatted_msg = f"{emoji} {msg}" if emoji else msg
        return colored(formatted_msg, color) if color else formatted_msg
    
    def log_with_format(
        self,
        level: int,
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False,
        *args,
        **kwargs
    ):
        """Helper method untuk logging dengan format."""
        formatted_msg = self._format_message(
            emoji_key, msg, color, highlight_values
        )
        super().log(level, formatted_msg, *args, **kwargs)
    
    def start(self, msg: str, *args, **kwargs):
        """Log start event dengan rocket emoji."""
        self.log_with_format(
            logging.INFO, 'start', msg, 'cyan', 
            *args, **kwargs
        )
    
    def success(self, msg: str, *args, **kwargs):
        """Log success event dengan checkmark emoji."""
        self.log_with_format(
            logging.INFO, 'success', msg, 'green',
            *args, **kwargs
        )
    
    def error(self, msg: str, *args, **kwargs):
        """Log error dengan X emoji."""
        self.log_with_format(
            logging.ERROR, 'error', msg, 'red',
            *args, **kwargs
        )
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning dengan warning emoji."""
        self.log_with_format(
            logging.WARNING, 'warning', msg, 'yellow',
            *args, **kwargs
        )
    
    def info(self, msg: str, *args, **kwargs):
        """Log info dengan info emoji."""
        self.log_with_format(
            logging.INFO, 'info', msg, None,
            *args, **kwargs
        )
    
    def metric(self, msg: str, *args, **kwargs):
        """Log metrics dengan chart emoji dan highlighted numbers."""
        self.log_with_format(
            logging.INFO, 'metric', msg, None,
            highlight_values=True, *args, **kwargs
        )
    
    def data(self, msg: str, *args, **kwargs):
        """Log data related info dengan clipboard emoji."""
        self.log_with_format(
            logging.INFO, 'data', msg, None,
            *args, **kwargs
        )
    
    def model(self, msg: str, *args, **kwargs):
        """Log model related info dengan robot emoji."""
        self.log_with_format(
            logging.INFO, 'model', msg, None,
            *args, **kwargs
        )
    
    def time(self, msg: str, *args, **kwargs):
        """Log timing info dengan timer emoji."""
        self.log_with_format(
            logging.INFO, 'time', msg, None,
            highlight_values=True, *args, **kwargs
        )

# Fungsi bantuan untuk mendapatkan logger dengan konfigurasi standar
def get_logger(
    name: str, 
    log_to_console: bool = True, 
    log_to_file: bool = True
) -> SmartCashLogger:
    """
    Get atau buat instance SmartCashLogger.
    
    Args:
        name: Nama logger yang diinginkan
        log_to_console: Flag untuk mengaktifkan logging ke konsol
        log_to_file: Flag untuk mengaktifkan logging ke file
        
    Returns:
        Instance SmartCashLogger
    """
    # Register SmartCashLogger sebagai logger class
    if not logging.getLoggerClass() == SmartCashLogger:
        logging.setLoggerClass(SmartCashLogger)
    
    # Get atau buat logger dengan konfigurasi standar
    logger = logging.getLogger(name)
    
    # Pastikan logger adalah SmartCashLogger dan memiliki handlers yang sesuai
    if isinstance(logger, SmartCashLogger) and not logger.handlers:
        # Reinisialisasi logger dengan parameter yang diberikan
        logger = SmartCashLogger(
            name=name,
            log_to_console=log_to_console,
            log_to_file=log_to_file
        )
    
    return logger