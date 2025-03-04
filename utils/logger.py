# File: smartcash/utils/logger.py
# Author: Alfrida Sabar
# Deskripsi: Logger yang kompatibel dengan Google Colab dengan dukungan emoji dan tampilan berwarna

import logging
import sys
import os
from typing import Optional
from datetime import datetime
from pathlib import Path
from IPython.display import display, HTML
import time

# Fungsi untuk menambahkan warna ke teks
def colored_text(text, color=None, weight="normal"):
    """Tambah warna ke teks menggunakan HTML"""
    if color:
        return f'<span style="color:{color}; font-weight:{weight}">{text}</span>'
    return text

class SmartCashLogger:
    """Logger yang kompatibel dengan Google Colab dengan dukungan emoji dan warna"""
    
    # Emoji untuk konteks log yang berbeda
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
    
    # Warna untuk konteks log yang berbeda
    COLORS = {
        'start': '#1e88e5',     # biru
        'success': '#43a047',   # hijau
        'error': '#e53935',     # merah
        'warning': '#ff9800',   # oranye
        'info': '#757575',      # abu-abu
        'data': '#7b1fa2',      # ungu
        'model': '#0097a7',     # cyan
        'time': '#6d4c41',      # coklat
        'metric': '#00897b',    # teal
        'save': '#00acc1',      # cyan terang
        'load': '#26a69a'       # teal terang
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
            log_to_colab: Flag untuk mengaktifkan logging ke Colab display
            log_to_file: Flag untuk mengaktifkan logging ke file
            log_dir: Direktori untuk menyimpan file log
        """
        self.name = name
        self.log_to_colab = log_to_colab
        self.log_to_file = log_to_file
        
        # Buat direktori log jika belum ada
        self.log_dir = Path(log_dir)
        if log_to_file:
            self.log_dir.mkdir(exist_ok=True, parents=True)
            
            # Setup file logger
            self.file_logger = logging.getLogger(name)
            self.file_logger.setLevel(logging.INFO)
            
            # Set file handler dengan rotasi harian
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"smartcash_{today}.log"
            
            # Cek handler yang sudah ada dan hapus jika perlu
            for handler in self.file_logger.handlers[:]:
                self.file_logger.removeHandler(handler)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)
    
    def _format_message(
        self, 
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False
    ) -> str:
        """Format pesan dengan emoji."""
        emoji = self.EMOJIS.get(emoji_key, '')
        
        # Format untuk file logger - tanpa HTML
        plain_msg = f"{emoji} {msg}" if emoji else msg
        
        # Jika highlight_values True, cari angka dalam teks untuk di-highlight
        if highlight_values and self.log_to_colab:
            words = msg.split()
            for i, word in enumerate(words):
                if any(char.isdigit() for char in word):
                    words[i] = colored_text(word, "#e65100", "bold")  # Oranye tua & bold
            
            formatted_msg = " ".join(words)
        else:
            formatted_msg = msg
        
        # Format untuk Colab display - dengan HTML
        if color and self.log_to_colab:
            html_msg = f"{emoji} {colored_text(formatted_msg, color)}" if emoji else colored_text(formatted_msg, color)
        else:
            html_msg = f"{emoji} {formatted_msg}" if emoji else formatted_msg
        
        return plain_msg, html_msg
    
    def log(
        self,
        level: str,
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False
    ):
        """Helper method untuk logging dengan format."""
        plain_msg, html_msg = self._format_message(
            emoji_key, msg, color, highlight_values
        )
        
        # Waktu untuk timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Log ke file jika diaktifkan
        if self.log_to_file:
            if level == 'info':
                self.file_logger.info(plain_msg)
            elif level == 'warning':
                self.file_logger.warning(plain_msg)
            elif level == 'error':
                self.file_logger.error(plain_msg)
            elif level == 'critical':
                self.file_logger.critical(plain_msg)
            else:
                self.file_logger.info(plain_msg)
        
        # Tampilkan di Colab jika diaktifkan
        if self.log_to_colab:
            display(HTML(f"<div>[{timestamp}] {html_msg}</div>"))
    
    def start(self, msg: str):
        """Log start event dengan rocket emoji."""
        self.log('info', 'start', msg, self.COLORS['start'])
    
    def success(self, msg: str):
        """Log success event dengan checkmark emoji."""
        self.log('info', 'success', msg, self.COLORS['success'])
    
    def error(self, msg: str):
        """Log error dengan X emoji."""
        self.log('error', 'error', msg, self.COLORS['error'])
    
    def warning(self, msg: str):
        """Log warning dengan warning emoji."""
        self.log('warning', 'warning', msg, self.COLORS['warning'])
    
    def info(self, msg: str):
        """Log info dengan info emoji."""
        self.log('info', 'info', msg, self.COLORS['info'])
    
    def metric(self, msg: str):
        """Log metrics dengan chart emoji dan highlighted numbers."""
        self.log('info', 'metric', msg, self.COLORS['metric'], highlight_values=True)
    
    def data(self, msg: str):
        """Log data related info dengan clipboard emoji."""
        self.log('info', 'data', msg, self.COLORS['data'])
    
    def model(self, msg: str):
        """Log model related info dengan robot emoji."""
        self.log('info', 'model', msg, self.COLORS['model'])
    
    def time(self, msg: str):
        """Log timing info dengan timer emoji."""
        self.log('info', 'time', msg, self.COLORS['time'], highlight_values=True)
        
    def progress(self, iterable=None, total=None, desc="Processing", **kwargs):
        """Buat progress bar yang kompatibel dengan Colab."""
        try:
            from tqdm.notebook import tqdm
            return tqdm(iterable, total=total, desc=desc, **kwargs)
        except ImportError:
            from tqdm.auto import tqdm
            return tqdm(iterable, total=total, desc=desc, **kwargs)

# Fungsi bantuan untuk mendapatkan logger dengan konfigurasi standar
def get_logger(
    name: str, 
    log_to_console: bool = True, 
    log_to_file: bool = True,
    log_to_colab: bool = False
) -> SmartCashLogger:
    """
    Get atau buat instance SmartCashLogger.
    
    Args:
        name: Nama logger yang diinginkan
        log_to_console: Flag untuk mengaktifkan logging ke konsol
        log_to_file: Flag untuk mengaktifkan logging ke file
        log_to_colab: Flag khusus untuk Google Colab (sama dengan log_to_console)
        
    Returns:
        Instance SmartCashLogger
    """
    # Register SmartCashLogger sebagai logger class
    if not logging.getLoggerClass() == SmartCashLogger:
        logging.setLoggerClass(SmartCashLogger)
    
    # Get atau buat logger dengan konfigurasi standar
    logger = logging.getLogger(name)
    
    # Gunakan log_to_colab sebagai alias untuk log_to_console jika diberikan
    if log_to_colab:
        log_to_console = True
    
    # Pastikan logger adalah SmartCashLogger dan memiliki handlers yang sesuai
    if isinstance(logger, SmartCashLogger) and not logger.handlers:
        # Reinisialisasi logger dengan parameter yang diberikan
        logger = SmartCashLogger(
            name=name,
            log_to_console=log_to_console,
            log_to_file=log_to_file
        )
    
    return logger