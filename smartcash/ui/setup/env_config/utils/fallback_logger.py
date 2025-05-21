"""
File: smartcash/ui/setup/env_config/utils/fallback_logger.py
Deskripsi: Logger fallback sederhana untuk environment config

DEPRECATED: Gunakan create_ui_logger dengan ENV_CONFIG_LOGGER_NAMESPACE dari smartcash.ui.utils.ui_logger_namespace sebagai gantinya.
"""

import logging
import sys
import warnings
from typing import Optional, Any

from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

# Tampilkan warning tentang deprecation
warnings.warn(
    "FallbackLogger sudah deprecated. Gunakan create_ui_logger dengan ENV_CONFIG_LOGGER_NAMESPACE "
    "dari smartcash.ui.utils.ui_logger_namespace sebagai gantinya.",
    DeprecationWarning,
    stacklevel=2
)

class FallbackLogger:
    """
    Logger sederhana yang menyediakan antarmuka mirip dengan UILogger
    untuk memastikan kompatibilitas ketika UILogger tidak tersedia.
    
    DEPRECATED: Gunakan create_ui_logger dengan ENV_CONFIG_LOGGER_NAMESPACE sebagai gantinya.
    """
    
    def __init__(self, name: str = "env_config", level: int = logging.INFO):
        """
        Inisialisasi logger
        
        Args:
            name: Nama logger
            level: Level logging
        """
        warnings.warn(
            "FallbackLogger sudah deprecated. Gunakan create_ui_logger dengan ENV_CONFIG_LOGGER_NAMESPACE sebagai gantinya.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Gunakan logger dengan namespace dari ui_logger_namespace sebagai gantinya
        self.logger = get_logger(ENV_CONFIG_LOGGER_NAMESPACE)
        self.logger.setLevel(level)
    
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
    
    DEPRECATED: Gunakan get_logger(ENV_CONFIG_LOGGER_NAMESPACE) sebagai gantinya.
    
    Args:
        name: Nama logger
        
    Returns:
        Instance FallbackLogger
    """
    warnings.warn(
        "get_fallback_logger sudah deprecated. Gunakan get_logger dengan ENV_CONFIG_LOGGER_NAMESPACE sebagai gantinya.",
        DeprecationWarning,
        stacklevel=2
    )
    return FallbackLogger(name) 