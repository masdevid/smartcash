# smartcash/ui/core/initializers/operation_initializer.py
"""
Operation initializer class for initializing operation-specific UI components.
Extends BaseInitializer with operation-specific capabilities.
"""
from typing import Dict, Any, Optional, Callable, List, Union, Type
import logging
import importlib

from smartcash.ui.core.initializers.base_initializer import BaseInitializer
from smartcash.ui.core.shared.logger import get_ui_logger
from smartcash.ui.decorators import safe_ui_operation


class OperationInitializer(BaseInitializer):
    """
    Initializer for operation-specific UI components.
    
    This class extends BaseInitializer with operation-specific capabilities,
    including progress tracking, dialog management, and summary generation.
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: str,
        operation_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the operation initializer.
        
        Args:
            module_name: Name of the module being initialized
            parent_module: Parent module name (e.g., 'dataset', 'setup')
            operation_name: Name of the operation
            logger: Optional logger instance
        """
        super().__init__(module_name, parent_module, logger)
        self.operation_name = operation_name
        
        # Operation components
        self.progress_tracker = None
        self.dialog = None
        self.summary_generator = None
        self.operation_handler = None
        
        # Ensure UI components include log output/accordion if not already present
        if 'log_output' not in self.ui_components:
            self.ui_components['log_output'] = None
        if 'log_accordion' not in self.ui_components:
            self.ui_components['log_accordion'] = None
    
    @safe_ui_operation(operation_name="setup_handlers", log_level="error")
    def setup_handlers(self) -> Dict[str, Any]:
        """
        Set up handlers for UI components.
        
        Returns:
            Dict with setup status and any error information
        """
        try:
            # Set up progress tracker
            self._setup_progress_tracker()
            
            # Set up dialog
            self._setup_dialog()
            
            # Set up summary generator
            self._setup_summary_generator()
            
            # Set up operation handler
            self.operation_handler = self._create_operation_handler()
            
            # Set up additional handlers
            self._setup_additional_handlers()
            
            return {
                "status": True,  # Using 'status' instead of 'success' for consistency
                "initializer": self.__class__.__name__
            }
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "initializer": self.__class__.__name__
            }
    
    def _setup_progress_tracker(self) -> None:
        """
        Set up the progress tracker for this operation.
        
        This method can be overridden by subclasses to set up
        a custom progress tracker.
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    def _setup_dialog(self) -> None:
        """
        Set up the dialog for this operation.
        
        This method can be overridden by subclasses to set up
        a custom dialog.
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    def _setup_summary_generator(self) -> None:
        """
        Set up the summary generator for this operation.
        
        This method can be overridden by subclasses to set up
        a custom summary generator.
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    def _create_operation_handler(self) -> Any:
        """
        Create the operation handler for this operation.
        
        Returns:
            Operation handler instance
        """
        # Try to dynamically load the operation-specific handler class
        try:
            handler_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.operations.{self.operation_name}_handler"
            handler_module = importlib.import_module(handler_path)
            
            # Look for a class named [OperationName]Handler (e.g., BatchDownloadHandler)
            handler_class_name = f"{self.operation_name.capitalize()}Handler"
            if hasattr(handler_module, handler_class_name):
                handler_class = getattr(handler_module, handler_class_name)
                return handler_class(
                    self.ui_components,
                    self.operation_name,
                    self.parent_module,
                    self.module_name,
                    self.logger
                )
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not load operation-specific handler: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error creating operation handler: {str(e)}")
        
        # Return None if no handler could be created
        return None
    
    def _setup_additional_handlers(self) -> None:
        """
        Set up additional handlers for this operation.
        
        This method can be overridden by subclasses to set up
        any additional handlers needed for the operation.
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    @safe_ui_operation(operation_name="run_operation", log_level="error")
    def run_operation(self, **kwargs) -> Dict[str, Any]:
        """
        Run the operation.
        
        Args:
            **kwargs: Additional arguments for the operation
            
        Returns:
            Dict with operation status and results
        """
        if not self.operation_handler:
            return {
                "status": False,
                "error": f"Operation handler for '{self.operation_name}' not initialized",
                "initializer": self.__class__.__name__
            }
        
        return self.operation_handler.run(**kwargs)
    
    @safe_ui_operation(operation_name="cancel_operation", log_level="error")
    def cancel_operation(self) -> Dict[str, Any]:
        """
        Cancel the operation.
        
        Returns:
            Dict with operation status
        """
        if not self.operation_handler:
            return {
                "status": False,
                "error": f"Operation handler for '{self.operation_name}' not initialized",
                "initializer": self.__class__.__name__
            }
        
        return self.operation_handler.cancel()
    
    @safe_ui_operation(operation_name="get_operation_status", log_level="debug")
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get the current status of the operation.
        
        Returns:
            Dict with operation status
        """
        if not self.operation_handler:
            return {
                "is_running": False,
                "is_completed": False,
                "is_cancelled": False,
                "has_error": True,
                "error": f"Operation handler for '{self.operation_name}' not initialized",
                "operation_name": self.operation_name,
                "initializer": self.__class__.__name__
            }
        
        return self.operation_handler.get_status()
    
    @safe_ui_operation(operation_name="get_operation_summary", log_level="debug")
    def get_operation_summary(self) -> Dict[str, Any]:
        """
        Get the summary of the operation.
        
        Returns:
            Dict with operation summary
        """
        if not self.operation_handler:
            return {
                "status": False,
                "error": f"Operation handler for '{self.operation_name}' not initialized",
                "initializer": self.__class__.__name__
            }
        
        return self.operation_handler.get_summary()
