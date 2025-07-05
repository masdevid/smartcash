"""Mock error handler module for testing."""
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ErrorHandler:
    """Mock error handler class."""
    
    def handle_error(self, 
                   error: Exception, 
                   context: Optional[Dict[str, Any]] = None,
                   show_to_user: bool = True,
                   log_level: str = "error") -> Dict[str, Any]:
        """Mock handle_error method."""
        return {
            "success": False,
            "error": str(error),
            "type": type(error).__name__,
            "context": context or {},
            "show_to_user": show_to_user,
            "log_level": log_level
        }

# Create a default instance for convenience
default_error_handler = ErrorHandler()
