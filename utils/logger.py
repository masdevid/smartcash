# File: utils/logger.py
# Author: Alfrida Sabar
# Deskripsi: Custom logger dengan emoji dan warna untuk monitoring eksperimen

import logging
import sys
from typing import Optional
from termcolor import colored

class SmartCashLogger:
    """Custom logger untuk SmartCash project dengan emoji dan colored output"""
    
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
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Console handler dengan colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler jika diperlukan
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _format_message(
        self, 
        emoji_key: str,
        msg: str,
        color: Optional[str] = None,
        highlight_values: bool = False
    ) -> str:
        """Format pesan dengan emoji dan warna"""
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
    
    def start(self, msg: str):
        """Log start event dengan rocket emoji"""
        self.logger.info(self._format_message('start', msg, 'cyan'))
    
    def success(self, msg: str):
        """Log success event dengan checkmark emoji"""
        self.logger.info(self._format_message('success', msg, 'green'))
    
    def error(self, msg: str):
        """Log error dengan X emoji"""
        self.logger.error(self._format_message('error', msg, 'red'))
    
    def warning(self, msg: str):
        """Log warning dengan warning emoji"""
        self.logger.warning(self._format_message('warning', msg, 'yellow'))
    
    def info(self, msg: str):
        """Log info dengan info emoji"""
        self.logger.info(self._format_message('info', msg))
    
    def metric(self, msg: str):
        """Log metrics dengan chart emoji dan highlighted numbers"""
        self.logger.info(self._format_message('metric', msg, highlight_values=True))
    
    def data(self, msg: str):
        """Log data related info dengan clipboard emoji"""
        self.logger.info(self._format_message('data', msg))
    
    def model(self, msg: str):
        """Log model related info dengan robot emoji"""
        self.logger.info(self._format_message('model', msg))
    
    def time(self, msg: str):
        """Log timing info dengan timer emoji"""
        self.logger.info(self._format_message('time', msg, highlight_values=True))