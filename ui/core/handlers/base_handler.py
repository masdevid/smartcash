# smartcash/ui/core/handlers/base_handler.py
"""
Base handler class for all UI handlers in SmartCash.
Provides common functionality and interface for all handlers.
"""
from typing import Dict, Any, Optional, Callable
import logging

from smartcash.ui.core.shared.logger import get_ui_logger
from smartcash.ui.core.shared.error_handler import get_ui_error_handler
from smartcash.ui.core.shared.ui_component_manager import get_ui_component_manager


class BaseHandler:
    """
    Base handler class for all UI handlers.
    
    This class provides common functionality and interface for all handlers
    in the SmartCash UI system. It handles basic initialization, logging,
    and error handling.
    """
    
    def __init__(
        self,
        ui_components: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base handler.
        
        Args:
            ui_components: Dictionary containing UI components
            logger: Optional logger instance, will create one if not provided
        """
        self.ui_components = ui_components
        
        # Set up centralized logger and error handler
        self.logger = logger or get_ui_logger(
            ui_components=ui_components,
            # Auto-determine suppression based on log_output readiness
            suppress_logs=None
        )
        self.error_handler = get_ui_error_handler(self.logger)
        
        # Initialize UI component manager
        self.ui_component_manager = get_ui_component_manager(ui_components, self.logger)
        
        self.setup()
        
    def setup(self) -> None:
        """
        Set up the handler after initialization.
        
        This method is called during initialization and can be overridden
        by subclasses to perform additional setup tasks.
        """
        pass
    
    def handle_error(self, error: Exception, context: str = "", show_traceback: bool = True, create_fallback_ui: bool = True) -> Dict[str, Any]:
        """
        Handle errors that occur during handler operations.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
            show_traceback: Whether to show the traceback in the logs
            create_fallback_ui: Whether to create fallback UI components
            
        Returns:
            Dict with error information
        """
        # Use centralized error handler
        result = self.error_handler.handle_error(
            error=error,
            context=context,
            show_traceback=show_traceback,
            create_fallback_ui=create_fallback_ui
        )
        
        # Add handler information
        result["handler"] = self.__class__.__name__
        
        return result
