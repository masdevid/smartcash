"""
File: smartcash/ui/dataset/preprocess/operations/base_preprocess_operation.py
Description: Base operation class for preprocessing operations
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from abc import ABC, abstractmethod

from smartcash.ui.logger import get_module_logger


class BasePreprocessOperation(ABC):
    """
    Base class for preprocessing operations.
    
    Features:
    - 📊 Progress tracking with callbacks
    - 📝 Logging with structured messages
    - 🔄 Async operation support
    - 🚨 Error handling
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any],
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None):
        """
        Initialize base preprocessing operation.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            progress_callback: Progress callback function
            log_callback: Log callback function
        """
        self.ui_components = ui_components
        self.config = config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        
        # Setup logging
        self.logger = get_module_logger(f"preprocess.{self.__class__.__name__.lower()}")
        
        # Operation state
        self.is_running = False
        self.current_progress = 0
        self.status_message = ""
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """
        Update progress with callback.
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress message
        """
        self.current_progress = progress
        self.status_message = message
        
        if self.progress_callback:
            try:
                self.progress_callback('overall', int(progress), 100, message)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
    
    def log_message(self, level: str, message: str) -> None:
        """
        Log message with callback.
        
        Args:
            level: Log level (info, success, warning, error)
            message: Log message
        """
        # Log to logger
        if level == 'info':
            self.logger.info(message)
        elif level == 'success':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)
        
        # Send to UI callback
        if self.log_callback:
            try:
                self.log_callback(level, message)
            except Exception as e:
                self.logger.error(f"Log callback error: {e}")
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.log_message('info', message)
    
    def log_success(self, message: str) -> None:
        """Log success message."""
        self.log_message('success', message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.log_message('warning', message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.log_message('error', message)
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the operation.
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Operation results dictionary
        """
        pass
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler."""
        return {
            'execute': self.execute
        }