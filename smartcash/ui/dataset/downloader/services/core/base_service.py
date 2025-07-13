"""
File: smartcash/ui/dataset/downloader/services/core/base_service.py
Description: Base service class for downloader services with common functionality.
"""

import logging
from typing import Any, Dict, Optional

class BaseService:
    """Base class for downloader services with common functionality.
    
    Features:
    - Standardized logging
    - Error handling
    - Common utility methods
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize base service with optional logger.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger will be used.
        """
        self.logger = logger or logging.getLogger(self.__class__.__module__)
    
    def log_info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message.
        
        Args:
            message: Error message to log
            exc_info: Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Warning message to log
        """
        self.logger.warning(message)
    
    def _handle_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Handle errors consistently across services.
        
        Args:
            message: Error message
            error: Optional exception that caused the error
        """
        if error:
            self.log_error(f"{message}: {str(error)}", exc_info=True)
        else:
            self.log_error(message)
