# smartcash/ui/core/shared/error_handler.py
"""
Centralized error handler for SmartCash UI components.
Provides standardized error handling functionality with UI integration.
"""
from typing import Dict, Any, Optional, Union, Type, Callable
import logging
import os
import sys
import traceback
from IPython.display import display

from smartcash.ui.core.shared.logger import UILogger, get_ui_logger
from smartcash.ui.components.error.error_component import create_error_response


class UIErrorHandler:
    """
    Centralized error handler for SmartCash UI components.
    
    This class provides standardized error handling functionality with UI integration.
    It ensures that errors are properly logged, displayed in the UI, and handled
    consistently across all SmartCash UI components.
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: str = "ui",
        ui_components: Optional[Dict[str, Any]] = None,
        logger: Optional[UILogger] = None
    ):
        """
        Initialize the UI error handler.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
            ui_components: Dictionary containing UI components
            logger: Optional UILogger instance
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.ui_components = ui_components or {}
        
        # Set up logger
        self.logger = logger or get_ui_logger(module_name, parent_module, ui_components, "error")
    
    def update_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Update UI components.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.ui_components = ui_components
        self.logger.update_ui_components(ui_components)
    
    def handle_error(
        self,
        error: Exception,
        context: str = "",
        show_traceback: bool = True,
        create_fallback_ui: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an error.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
            show_traceback: Whether to show the traceback in the logs
            create_fallback_ui: Whether to create fallback UI components
            
        Returns:
            Dict with error information
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log the error
        if context:
            log_message = f"{context}: {error_type} - {error_message}"
        else:
            log_message = f"{error_type} - {error_message}"
        
        if show_traceback:
            self.logger.exception(log_message)
        else:
            self.logger.error(log_message)
        
        # Create fallback UI if requested
        if create_fallback_ui:
            self._create_fallback_ui(error, context)
        
        # Return error information
        return {
            "status": False,  # Using 'status' instead of 'success' for consistency
            "error": error_message,
            "error_type": error_type,
            "context": context,
            "module": self.module_name
        }
    
    def _create_fallback_ui(self, error: Exception, context: str = "") -> None:
        """
        Create fallback UI components.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
        """
        if not self.ui_components:
            return
        
        try:
            # Import error component directly from the component module
            from smartcash.ui.components.error.error_component import create_error_component
            
            # Create error component with traceback
            error_component = create_error_component(
                error_message=str(error),
                traceback=traceback.format_exc() if hasattr(traceback, 'format_exc') else None,
                title=f"🚨 Error in {self.module_name}",
                error_type="error",
                show_traceback=True
            )
            
            # Display the error component
            if 'widget' in error_component:
                display(error_component['widget'])
            else:
                display(error_component)
                
        except Exception as e:
            # If we can't create fallback UI, just log the error
            self.logger.error(f"Failed to create fallback UI: {str(e)}")
    
    def create_error_result(
        self,
        message: str,
        error_type: str = "Error",
        context: str = "",
        create_fallback_ui: bool = False
    ) -> Dict[str, Any]:
        """
        Create an error result without an actual exception.
        
        Args:
            message: Error message
            error_type: Type of error
            context: Additional context about where the error occurred
            create_fallback_ui: Whether to create fallback UI components
            
        Returns:
            Dict with error information
        """
        # Log the error
        if context:
            log_message = f"{context}: {error_type} - {message}"
        else:
            log_message = f"{error_type} - {message}"
        
        self.logger.error(log_message)
        
        # Create fallback UI if requested
        if create_fallback_ui:
            try:
                error = Exception(message)
                self._create_fallback_ui(error, context)
            except Exception:
                pass
        
        # Return error information
        return {
            "status": False,  # Using 'status' instead of 'success' for consistency
            "error": message,
            "error_type": error_type,
            "context": context,
            "module": self.module_name
        }


def get_ui_error_handler(
    module_name: str,
    parent_module: str = "ui",
    ui_components: Optional[Dict[str, Any]] = None,
    logger: Optional[UILogger] = None
) -> UIErrorHandler:
    """
    Get a UI error handler instance.
    
    Args:
        module_name: Name of the module
        parent_module: Parent module name
        ui_components: Dictionary containing UI components
        logger: Optional UILogger instance
        
    Returns:
        UIErrorHandler instance
    """
    return UIErrorHandler(module_name, parent_module, ui_components, logger)
