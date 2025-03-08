"""
File: smartcash/utils/simple_logger.py
Author: Alfrida Sabar
Deskripsi: Logger sederhana yang dapat digunakan sebagai fallback jika SmartCashLogger
tidak tersedia. Menyediakan fungsionalitas dasar dengan emoji dan warna.
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
from termcolor import colored

class SimpleLogger:
    """Logger fallback sederhana dengan dukungan emoji dan warna."""
    
    # Emoji untuk setiap level log
    LEVEL_EMOJI = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥',
        'SUCCESS': '✅',
        'START': '🔄'
    }
    
    # Warna untuk setiap level log
    LEVEL_COLOR = {
        'DEBUG': 'grey',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
        'SUCCESS': 'green',
        'START': 'blue'
    }
    
    def __init__(self, name: str = "smartcash", level: int = logging.INFO):
        """
        Inisialisasi SimpleLogger.
        
        Args:
            name: Nama logger
            level: Level logging (default: INFO)
        """
        self.name = name
        self.level = level
        
        # Setup logger dasar
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Setup handler jika belum ada
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # Format yang lebih sederhana
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            
            # File handler dengan rotasi
            try:
                log_file = f"logs/{name}_{datetime.now().strftime('%Y-%m-%d')}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except:
                # Jika gagal membuat file log, lanjutkan tanpa file handler
                pass
    
    def _format_message(self, level: str, message: str) -> str:
        """
        Format pesan log dengan emoji dan warna sesuai level.
        
        Args:
            level: Level log
            message: Pesan log
            
        Returns:
            Pesan yang telah diformat
        """
        emoji = self.LEVEL_EMOJI.get(level, '')
        color = self.LEVEL_COLOR.get(level, 'white')
        
        try:
            return f"{emoji} {colored(message, color)}"
        except:
            # Jika termcolor gagal, gunakan pesan tanpa warna
            return f"{emoji} {message}"
    
    def debug(self, message: str) -> None:
        """Log pesan level DEBUG."""
        self.logger.debug(self._format_message('DEBUG', message))
    
    def info(self, message: str) -> None:
        """Log pesan level INFO."""
        self.logger.info(self._format_message('INFO', message))
    
    def warning(self, message: str) -> None:
        """Log pesan level WARNING."""
        self.logger.warning(self._format_message('WARNING', message))
    
    def error(self, message: str) -> None:
        """Log pesan level ERROR."""
        self.logger.error(self._format_message('ERROR', message))
    
    def critical(self, message: str) -> None:
        """Log pesan level CRITICAL."""
        self.logger.critical(self._format_message('CRITICAL', message))
    
    def success(self, message: str) -> None:
        """Log pesan sukses (level INFO)."""
        self.logger.info(self._format_message('SUCCESS', message))
    
    def start(self, message: str) -> None:
        """Log pesan awal proses (level INFO)."""
        self.logger.info(self._format_message('START', message))
    
    def __call__(self, message: str) -> None:
        """Menggunakan logger sebagai callable untuk interface yang sama dengan print."""
        self.info(message)