"""
File: smartcash/utils/logger.py
Author: Alfrida Sabar
Deskripsi: Logger terintegrasi untuk SmartCash yang mendukung berbagai lingkungan dengan emojis, 
           warna, dan berbagai output target (file, console, Colab).
"""

import logging
import sys
import os
from typing import Optional, Tuple, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import threading

# Support untuk tampilan Colab
try:
    from IPython.display import display, HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Support untuk text berwarna di terminal
try:
    from termcolor import colored
    HAS_TERMCOLOR = True
except ImportError:
    HAS_TERMCOLOR = False


class SmartCashLogger:
    """
    Logger yang fleksibel dengan dukungan untuk:
    - Output ke file, konsol, dan Colab
    - Emojis untuk konteks log
    - Text berwarna untuk highlight
    - Thread safety
    """
    
    # Emoji untuk berbagai konteks log
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
        'load': '📂',
        'debug': '🐞',
        'config': '⚙️'
    }
    
    # Warna untuk berbagai konteks log
    COLORS = {
        'start': 'blue',
        'success': 'green',
        'error': 'red',
        'warning': 'yellow',
        'info': 'white',
        'data': 'magenta',
        'model': 'cyan',
        'time': 'yellow',
        'metric': 'green',
        'save': 'cyan',
        'load': 'cyan',
        'debug': 'grey',
        'config': 'blue'
    }
    
    # Lock untuk thread safety
    _lock = threading.RLock()
    
    def __init__(
        self, 
        name: str,
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_to_colab: bool = None,
        log_dir: str = "logs",
        use_colors: bool = True,
        use_emojis: bool = True
    ):
        """
        Inisialisasi logger.
        
        Args:
            name: Nama logger
            level: Level logging (default: INFO)
            log_to_file: Flag untuk aktivasi logging ke file
            log_to_console: Flag untuk aktivasi logging ke konsol
            log_to_colab: Flag untuk aktivasi logging ke Colab (None untuk deteksi otomatis)
            log_dir: Direktori untuk menyimpan file log
            use_colors: Gunakan warna dalam output
            use_emojis: Gunakan emoji dalam output
        """
        self.name = name
        self.level = level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.use_colors = use_colors and HAS_TERMCOLOR
        self.use_emojis = use_emojis
        
        # Deteksi Colab secara otomatis jika tidak ditentukan
        if log_to_colab is None:
            self.log_to_colab = self._detect_colab() and HAS_IPYTHON
        else:
            self.log_to_colab = log_to_colab and HAS_IPYTHON
        
        # Setup dasar logger
        self._setup_logger(log_dir)
    
    def _detect_colab(self) -> bool:
        """Deteksi apakah berjalan di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _setup_logger(self, log_dir: str) -> None:
        """
        Setup logger dasar.
        
        Args:
            log_dir: Direktori untuk file log
        """
        with self._lock:
            # Setup logger
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(self.level)
            
            # Hapus handler yang mungkin sudah ada
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            
            # Tambahkan console handler jika diminta
            if self.log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self.level)
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # Tambahkan file handler jika diminta
            if self.log_to_file:
                self.log_dir = Path(log_dir)
                self.log_dir.mkdir(exist_ok=True, parents=True)
                
                today = datetime.now().strftime("%Y-%m-%d")
                log_file = self.log_dir / f"smartcash_{today}.log"
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(self.level)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def _format_message(
        self, 
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False
    ) -> Tuple[str, str]:
        """
        Format pesan dengan emoji dan warna.
        
        Args:
            emoji_key: Kunci emoji dari dict EMOJIS
            msg: Pesan yang akan diformat
            color: Warna teks (opsional)
            highlight_values: Highlight nilai numerik
            
        Returns:
            Tuple (plain_message, html_message)
        """
        # Tentukan emoji yang akan digunakan
        emoji = self.EMOJIS.get(emoji_key, '') if self.use_emojis else ''
        
        # Format plain message untuk file logger
        plain_msg = f"{emoji} {msg}" if emoji else msg
        
        # Format untuk output HTML (Colab)
        if self.log_to_colab:
            # Highlight nilai numerik jika diminta
            if highlight_values:
                import re
                # Regex untuk mendeteksi angka
                num_pattern = r'\b\d+\.?\d*\b'
                # Highlight angka dengan warna orange
                msg = re.sub(num_pattern, 
                            r'<span style="color: #e65100; font-weight: bold;">\g<0></span>', 
                            msg)
            
            # Tambahkan warna ke teks jika ditentukan
            if color:
                html_msg = f"{emoji} <span style='color: {color};'>{msg}</span>" if emoji else f"<span style='color: {color};'>{msg}</span>"
            else:
                html_msg = f"{emoji} {msg}" if emoji else msg
        else:
            # Jika tidak menggunakan Colab, gunakan plain message
            html_msg = plain_msg
            
            # Tambahkan warna untuk terminal jika diminta
            if self.use_colors and color and HAS_TERMCOLOR:
                html_msg = colored(plain_msg, color)
        
        return plain_msg, html_msg
    
    def log(
        self,
        level: str,
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False
    ) -> None:
        """
        Log pesan dengan format yang sesuai.
        
        Args:
            level: Level log ('info', 'warning', etc.)
            emoji_key: Kunci emoji
            msg: Pesan log
            color: Warna untuk teks
            highlight_values: Highlight nilai numerik
        """
        with self._lock:
            plain_msg, html_msg = self._format_message(
                emoji_key, msg, color, highlight_values
            )
            
            # Waktu untuk timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Log ke file via logger
            if level == 'debug':
                self.logger.debug(plain_msg)
            elif level == 'info':
                self.logger.info(plain_msg)
            elif level == 'warning':
                self.logger.warning(plain_msg)
            elif level == 'error':
                self.logger.error(plain_msg)
            elif level == 'critical':
                self.logger.critical(plain_msg)
            
            # Tampilkan di Colab jika aktif
            if self.log_to_colab and HAS_IPYTHON:
                display(HTML(f"<div>[{timestamp}] {html_msg}</div>"))
    
    # Shortcut methods for common log levels
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.log('debug', 'debug', msg, self.COLORS.get('debug'))
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.log('info', 'info', msg, self.COLORS.get('info'))
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.log('warning', 'warning', msg, self.COLORS.get('warning'))
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.log('error', 'error', msg, self.COLORS.get('error'))
    
    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.log('critical', 'error', msg, self.COLORS.get('error'))
    
    def success(self, msg: str) -> None:
        """Log success message."""
        self.log('info', 'success', msg, self.COLORS.get('success'))
    
    def start(self, msg: str) -> None:
        """Log start message."""
        self.log('info', 'start', msg, self.COLORS.get('start'))
    
    def metric(self, msg: str) -> None:
        """Log metric with highlighted numbers."""
        self.log('info', 'metric', msg, self.COLORS.get('metric'), highlight_values=True)
    
    def data(self, msg: str) -> None:
        """Log data related message."""
        self.log('info', 'data', msg, self.COLORS.get('data'))
    
    def model(self, msg: str) -> None:
        """Log model related message."""
        self.log('info', 'model', msg, self.COLORS.get('model'))
    
    def time(self, msg: str) -> None:
        """Log timing information."""
        self.log('info', 'time', msg, self.COLORS.get('time'), highlight_values=True)
    
    def config(self, msg: str) -> None:
        """Log configuration information."""
        self.log('info', 'config', msg, self.COLORS.get('config'))
    
    def progress(self, iterable=None, total=None, desc="Processing", **kwargs):
        """Create a progress bar compatible with various environments."""
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, total=total, desc=desc, **kwargs)
        except ImportError:
            return iterable  # Fallback to normal iterator if tqdm not available


# Factory function untuk kemudahan penggunaan
def get_logger(
    name: str, 
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_to_colab: bool = None,
    log_dir: str = "logs"
) -> SmartCashLogger:
    """
    Fungsi factory untuk mendapatkan logger.
    
    Args:
        name: Nama logger
        level: Level logging
        log_to_file: Aktifkan logging ke file
        log_to_console: Aktifkan logging ke konsol
        log_to_colab: Aktifkan logging ke Colab (None untuk deteksi otomatis)
        log_dir: Direktori untuk file log
        
    Returns:
        Instance SmartCashLogger
    """
    return SmartCashLogger(
        name=name,
        level=level,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
        log_to_colab=log_to_colab,
        log_dir=log_dir
    )