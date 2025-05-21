"""
File: smartcash/ui/setup/env_config/utils/fallback_logger.py
Deskripsi: Logger fallback sederhana untuk environment config
"""

import logging
import sys
from typing import Optional, Any

class FallbackLogger:
    """
    Logger sederhana yang menyediakan antarmuka mirip dengan UILogger
    untuk memastikan kompatibilitas ketika UILogger tidak tersedia.
    """
    
    def __init__(self, name: str = "env_config", level: int = logging.INFO):
        """
        Inisialisasi logger
        
        Args:
            name: Nama logger
            level: Level logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Hapus handler yang sudah ada untuk mencegah duplikasi
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Tambahkan console handler ke sys.__stdout__ untuk menghindari rekursi
        handler = logging.StreamHandler(sys.__stdout__)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def success(self, message: str) -> None:
        """Log success message."""
        self.logger.info(f"✅ {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(f"❌ {message}")
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(f"🔥 {message}")
    
    def set_level(self, level: int) -> None:
        """Set log level."""
        self.logger.setLevel(level)

def get_fallback_logger(name: str = "env_config") -> FallbackLogger:
    """
    Dapatkan instance fallback logger
    
    Args:
        name: Nama logger
        
    Returns:
        Instance FallbackLogger
    """
    return FallbackLogger(name) 