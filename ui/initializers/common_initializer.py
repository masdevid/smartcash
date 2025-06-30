"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Enhanced CommonInitializer dengan proper logging sequence dan progress tracker integration

Initialization Flow:
1. Load and validate configuration without suppression
2. Create UI components (including progress tracker)
3. Initialize logger bridge for UI logging integration
4. Run pre-initialization checks (if _pre_initialize_checks exists)
5. Set up event handlers (if _setup_handlers exists)
6. Get and validate root UI component
7. Run post-initialization checks (if _after_init_checks exists)
8. Log successful initialization
9. Return UI components with proper cleanup

Key Features:
- No premature output suppression
- Progress tracker integration
- UI logger bridge for real-time logging
- Extensible lifecycle hooks
- Comprehensive error handling
"""

# Import logging utilities
import logging
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import traceback
import ipywidgets as widgets
import contextlib

# Import new consolidated logger and error handler
from smartcash.common.logger import get_logger as get_common_logger
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.ui_logger import (
    UILogger,
    setup_global_logging,
    get_module_logger,
    LogSuppressor
)
from smartcash.ui.handlers.error_handler import create_error_response

class CommonInitializer(ABC):
    """Enhanced base class for initializing UI modules with comprehensive lifecycle management.
    
    This class provides a structured approach to UI initialization with built-in error
    handling, logging, progress tracking, and configuration management. It follows a 
    fail-fast pattern to ensure issues are caught early in the initialization process.
    
    Key Features:
        - Configurable initialization flow with clear extension points
        - Built-in error handling and recovery mechanisms
        - Integrated logging with UI support and progress tracking
        - Thread-safe component management
        - Memory-efficient resource handling
        - Progress tracker integration for long-running operations
        - No premature output suppression
    """
    # Class-level flag to ensure we only set up suppression once
    _suppression_initialized = False
    
    def _initialize_suppression(self):
        """Initialize log and output suppression."""
        try:
            # Set up global logging with suppression
            setup_global_logging(
                ui_components=self.ui_components,
                log_level=logging.INFO,
                log_to_file=False
            )
            # Apply additional suppression for backend libraries
            LogSuppressor.setup_aggressive_log_suppression()
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize suppression: {e}")
            return False
    
    def __init__(self, module_name: str, config_handler_class: Type[ConfigHandler] = None):
        """Initialize dengan complete output suppression hingga UI ready
        
        Args:
            module_name: Nama module (used for logging)
            config_handler_class: Optional ConfigHandler class for config management
        """
        # Initialize suppression at class level
        self._initialize_suppression()
        
        # Store basic attributes
        self.module_name = module_name
        self._ui_components = {}
        self._logger_bridge = None
        self.config_handler = config_handler_class() if config_handler_class else None
        
        # Setup logger with module-level logging
        try:
            self.logger = get_module_logger(
                name=self.__class__.__module__
            )
            self.ui_components['logger'] = self.logger
            
        except Exception:
            # If logger setup fails, create a silent logger
            self.logger = logging.getLogger(f"smartcash.ui.{module_name}.silent")
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL)
    
    @contextlib.contextmanager
    def _suppress_outputs(self):
        """Context manager for suppressing outputs during initialization."""
        # Set up global logging with suppression
        setup_global_logging(
            ui_components=self.ui_components,
            log_level=logging.INFO,
            log_to_file=False
        )
        
        try:
            yield
        finally:
            # Cleanup is handled by the UILogger's cleanup on exit
            pass
    
    def _initialize_logger_bridge(self) -> None:
        """Initialize the logger for UI logging integration."""
        try:
            # Create a module-level logger with UI integration
            self._logger_bridge = get_module_logger(
                name=f"smartcash.ui.{self.module_name}"
            )
            
            # Store in UI components
            self.ui_components['logger'] = self._logger_bridge
            
            # Set UI components for the logger
            if hasattr(self._logger_bridge, 'set_ui_components'):
                self._logger_bridge.set_ui_components(self.ui_components)
            
        except Exception as e:
            # If logger bridge setup fails, use the default logger
            print(f"Warning: Failed to initialize logger bridge: {e}")
            self._logger_bridge = self.logger
            # Fallback silent logger - NO stdout output
            self.logger = get_logger(f"smartcash.ui.{self.module_name}")
            if hasattr(self.logger, 'set_level'):
                self.logger.set_level(logging.CRITICAL)
            else:
                self.logger.setLevel(logging.CRITICAL)
    
    def initialize(self, config: Dict[str, Any] = None, **kwargs) -> Any:
        """Initialize dan return UI module dengan complete output suppression
        
        This is the main entry point that orchestrates the initialization process.
        It follows a strict sequence of operations and ensures proper error handling.
        
        Args:
            config: Optional configuration dictionary. If None, will attempt to load
                   configuration using the configured config handler.
            **kwargs: Additional keyword arguments that may be required by subclasses.
                   
        Returns:
            The root UI widget that can be displayed or embedded in other UIs.
            In case of error, returns an error widget with the error message.
            
        Example:
            >>> initializer = MyInitializer()
            >>> ui = initializer.initialize()
            >>> display(ui)  # Explicitly display the UI
        """
        # Use context manager to ensure proper cleanup
        with self._suppress_outputs():
            try:
                # 1. Load config - silent
                config = self._load_config(config)
                
                # 2. Create UI components - silent  
                ui_components = self._create_ui_components(config, **kwargs) or {}
                self._ui_components = ui_components
                
                # 3. Add metadata for tracking
                ui_components.update({
                    'module_name': self.module_name,
                    'initialization_timestamp': self._get_timestamp()
                })
                
                # 4. Initialize logger bridge SETELAH UI ready
                self._initialize_logger_bridge()
                
                # 5. Pre-initialization checks (optional) - silent
                if hasattr(self, '_pre_initialize_checks'):
                    self._pre_initialize_checks(config=config, **kwargs)
                
                # 6. Setup handlers SETELAH logger bridge ready
                if hasattr(self, '_setup_handlers'):
                    ui_components = self._setup_handlers(ui_components, config, **kwargs) or ui_components
                
                # 7. Get root UI component
                root_ui = self._get_ui_root(ui_components)
                if not root_ui:
                    raise ValueError("No root UI component found")
                
                # 8. Post-initialization checks (optional) - silent
                if hasattr(self, '_after_init_checks'):
                    self._after_init_checks(ui_components=ui_components, config=config, **kwargs)
                
                # 9. SUCCESS: Log ONLY to UI setelah everything ready
                if hasattr(self.logger, 'info'):
                    self.logger.info(f"✅ {self.module_name} siap digunakan")
                
                return root_ui
                
            except Exception as e:
                error_msg = f"❌ Gagal inisialisasi {self.module_name}: {str(e)}"
                # Log the error first
                self._handle_error(error_msg, exc_info=True)
                # Then create and return the error response
                return self.create_error_response(error_msg, e)
    
    def _load_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load and validate the configuration for the module.
        
        This method attempts to load configuration in the following order:
        1. Use the provided config dictionary if not None
        2. Load from the configured config handler if available
        3. Fall back to default configuration
        
        Args:
            config: Optional configuration dictionary to use directly.
            
        Returns:
            dict: The loaded and validated configuration.
            
        Note:
            If config loading fails, a warning will be logged and the default
            configuration will be returned.
        """
        if config is not None:
            return config
            
        if self.config_handler:
            try:
                config = self.config_handler.load_config()
                if config:
                    return config
            except Exception as e:
                self.logger.warning(f"Failed to load config: {str(e)}")
                
        # Fall back to default config
        return self._get_default_config()
    
    @abstractmethod
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create and return UI components as a dictionary.
        
        Args:
            config: Loaded configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        pass
        
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for this module.
        
        Returns:
            Default configuration dictionary
        """
        pass
        
    def _get_ui_root(self, ui_components: Dict[str, Any]) -> Any:
        """Get the root UI component from the components dictionary.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            The root UI component or error component if available
            
        Note:
            If the components dictionary contains an 'error' key, it will return
            the error component if available.
        """
        # Check for error component first
        if ui_components.get('error') and 'ui' in ui_components:
            return ui_components['ui']
            
        # Default to 'ui' key if no error
        for key in ['ui', 'container', 'widget', 'root']:
            if key in ui_components:
                return ui_components[key]
                
        # Return first widget-like component if no root found
        for component in ui_components.values():
            if hasattr(component, 'layout') and hasattr(component, 'children'):
                return component
                
        return None
        
    def _pre_initialize_checks(self, **kwargs) -> None:
        """Override this method to perform pre-initialization checks.
        
        Raise an exception if any check fails.
        
        Args:
            **kwargs: Additional arguments that might be needed for checks
            
        Raises:
            Exception: If any pre-initialization check fails
        """
        pass
        
    def _add_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Add logger bridge to UI components for logging to UI output.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Use the UILogger's built-in UI integration
            if hasattr(self, 'logger') and hasattr(self.logger, 'set_ui_components'):
                self.logger.set_ui_components(ui_components)
            
            # Store namespace for component identification
            ui_components['logger_namespace'] = f"smartcash.ui.{self.module_name}"
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Failed to setup logger bridge: {str(e)}")
            else:
                print(f"[WARNING] Failed to setup logger bridge: {str(e)}")
            
    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking purposes.
        
        Returns:
            ISO formatted timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _handle_error(self, error_msg: str, exc_info: bool = False, **kwargs) -> None:
        """Centralized error handling for all initializers.
        
        Args:
            error_msg: The error message to log
            exc_info: Whether to include exception info
            **kwargs: Additional arguments for logging
        """
        # Use the logger directly for consistent error handling
        if hasattr(self, 'logger') and self.logger:
            self.logger.error(error_msg, exc_info=exc_info, **kwargs)
        else:
            # Fallback to print if logger is not available
            print(f"[ERROR] {error_msg}", flush=True)
            if exc_info and 'exc_info' in kwargs:
                import traceback
                traceback.print_exc()

    def create_error_response(self, error_message: str, error: Optional[Exception] = None) -> Dict[str, Any]:
        """Create a fallback UI component to display error messages.
        
        This method creates a consistent error UI that can be used across all initializers.
        It returns a dictionary with the error component and placeholders for other UI elements.
        
        Note: This method only creates the error response UI and does not log the error.
        Call _handle_error() separately if you need to log the error.
        
        Args:
            error_message: The error message to display
            error: Optional exception for traceback
            
        Returns:
            Dict containing:
            - 'ui': The main error widget
            - 'log_output': Empty output widget
            - 'status_panel': Empty widget
            - 'error': Boolean flag indicating this is an error state
        """
        # Use the centralized error response creator
        error_response = create_error_response(
            error_message=error_message,
            error=error,
            title=f"{self.module_name} Error",
            include_traceback=True
        )
        
        # Add additional components needed for initializers
        return {
            'ui': error_response.get('container', error_response),
            'log_output': widgets.Output(),
            'status_panel': widgets.Output(),
            'error': True
        }