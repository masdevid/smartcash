"""
Buffered logger utility for collecting log messages before the UI is fully initialized.
"""
from typing import Any, Optional, Callable

class BufferedLogger:
    """
    A logger that buffers messages until a UI logger is available.
    
    This is useful for collecting log messages during initialization
    before the UI components are fully set up.
    """
    
    def __init__(self):
        """Initialize the buffered logger with an empty buffer."""
        self.buffer = []
        self.ui_logger = None
        
    def log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a message with the specified level.
        
        If the UI logger is available, the message is logged immediately.
        Otherwise, it's buffered for later.
        
        Args:
            level: Log level (e.g., 'info', 'error')
            message: Log message
            *args: Positional arguments for the log method
            **kwargs: Keyword arguments for the log method
        """
        if self.ui_logger and hasattr(self.ui_logger, level):
            getattr(self.ui_logger, level)(message, *args, **kwargs)
        else:
            self.buffer.append((level, message, args, kwargs))
                
    def flush_to_ui_logger(self, ui_logger: Any) -> None:
        """
        Flush buffered messages to the provided UI logger.
        
        Args:
            ui_logger: The target logger that will receive buffered messages
        """
        self.ui_logger = ui_logger
        for level, message, args, kwargs in self.buffer:
            if hasattr(ui_logger, level):
                getattr(ui_logger, level)(message, *args, **kwargs)
        self.buffer = []
        
    def clear_buffer(self) -> None:
        """Clear all buffered messages."""
        self.buffer = []
        
    @staticmethod
    def clear_logs(logger: Any) -> bool:
        """Clear logs from any logger instance if it has a clear_buffer method.
        
        Args:
            logger: The logger instance to clear logs from
            
        Returns:
            bool: True if logs were cleared, False otherwise
        """
        try:
            if hasattr(logger, 'clear_buffer') and callable(logger.clear_buffer):
                logger.clear_buffer()
                return True
            return False
        except Exception:
            return False


def create_buffered_logger() -> BufferedLogger:
    """
    Create a new BufferedLogger instance with standard log methods.
    
    Returns:
        BufferedLogger: A logger instance with standard log methods
    """
    logger = BufferedLogger()
    # Add standard log methods
    for level in ['debug', 'info', 'warning', 'error', 'critical', 'success']:
        setattr(logger, level, 
               lambda msg, *a, lvl=level, **kw: logger.log(lvl, msg, *a, **kw))
    return logger
