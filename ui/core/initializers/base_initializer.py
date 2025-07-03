# smartcash/ui/core/initializers/base_initializer.py
"""
Base initializer class for all UI initializers in SmartCash.
Provides common functionality and lifecycle hooks for UI initialization.
"""
from typing import Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from smartcash.ui.core.shared.logger import get_ui_logger
from smartcash.ui.core.shared.error_handler import get_ui_error_handler
from smartcash.ui.core.shared.ui_component_manager import get_ui_component_manager
from smartcash.ui.decorators import safe_ui_operation


class BaseInitializer(ABC):
    """
    Base initializer class for all UI initializers.
    
    This class provides common functionality and lifecycle hooks for UI initialization
    in the SmartCash UI system. It handles basic initialization, logging,
    error handling, and defines the initialization lifecycle.
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base initializer.
        
        Args:
            module_name: Name of the module being initialized
            parent_module: Optional parent module name
            logger: Optional logger instance, will create one if not provided
        """
        self.module_name = module_name
        self.parent_module = parent_module
        
        # UI components dictionary
        self.ui_components = {}
        
        # Initialize UI component manager
        self.ui_component_manager = get_ui_component_manager(self.ui_components, self.logger)
        
        # Set up centralized logger and error handler
        log_path = f"{parent_module}.{module_name}" if parent_module else module_name
        self.logger = get_ui_logger(
            module_name=module_name,
            parent_module=f"ui.{parent_module}" if parent_module else "ui",
            ui_components=self.ui_components,
            # Auto-determine suppression based on log_output readiness
            suppress_logs=None
        )
        
        self.error_handler = get_ui_error_handler(
            module_name=module_name,
            parent_module=f"ui.{parent_module}" if parent_module else "ui",
            ui_components=self.ui_components,
            logger=self.logger
        )
        
        # Initialization state
        self.is_initialized = False
        self.has_error = False
        self.error_info = None
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the UI components.
        
        This method orchestrates the initialization lifecycle:
        1. Pre-initialization checks
        2. Create UI components
        3. Set up handlers
        4. Post-initialization checks
        
        Returns:
            Dictionary containing UI components or error information
        """
        try:
            self.logger.info(f"Initializing {self.module_name} module...")
            
            # Reset state
            self.is_initialized = False
            self.has_error = False
            self.error_info = None
            
            # 1. Pre-initialization checks
            if not self.pre_initialization_checks():
                self.logger.error(f"Pre-initialization checks failed for {self.module_name}")
                return self.ui_components
            
            # 2. Create UI components
            self.ui_components = self.create_ui_components()
            if not self.ui_components:
                self.logger.error(f"Failed to create UI components for {self.module_name}")
                return self.ui_components
            
            # 3. Set up handlers
            setup_result = self.setup_handlers()
            if not setup_result.get("status", False):
                self.logger.error(f"Failed to set up handlers for {self.module_name}")
                return self.ui_components
            
            # 4. Post-initialization checks
            if not self.post_initialization_checks():
                self.logger.error(f"Post-initialization checks failed for {self.module_name}")
                return self.ui_components
            
            # Mark as initialized
            self.is_initialized = True
            self.logger.info(f"{self.module_name} module initialized successfully")
            
            return self.ui_components
        except Exception as e:
            return self.handle_error(e, f"Error initializing {self.module_name} module")
    
    @safe_ui_operation(operation_name="pre_initialization_checks", log_level="error", fallback_return=False)
    def pre_initialization_checks(self) -> bool:
        """
        Perform checks before initialization.
        
        Returns:
            True if checks pass, False otherwise
        """
        # This method can be overridden by subclasses
        return True
    
    @abstractmethod
    def create_ui_components(self) -> Dict[str, Any]:
        """
        Create UI components.
        
        This method must be implemented by subclasses.
        
        Returns:
            Dictionary containing UI components
        """
        pass
    
    @abstractmethod
    def setup_handlers(self) -> Dict[str, Any]:
        """
        Set up handlers for UI components.
        
        This method must be implemented by subclasses.
        
        Returns:
            Dict with setup status and any error information
        """
        pass
    
    @safe_ui_operation(operation_name="post_initialization_checks", log_level="error", fallback_return=False)
    def post_initialization_checks(self) -> bool:
        """
        Perform checks after initialization.
        
        Returns:
            True if checks pass, False otherwise
        """
        # This method can be overridden by subclasses
        return True
    
    def handle_error(self, error: Exception, context: str = "", show_traceback: bool = True, create_fallback_ui: bool = True) -> Dict[str, Any]:
        """
        Handle errors that occur during initialization.
        
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
        
        # Update state
        self.has_error = True
        self.error_info = {
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context
        }
        
        # Add initializer information
        result["initializer"] = self.__class__.__name__
        
        return result
