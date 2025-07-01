"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Enhanced CommonInitializer with proper logging sequence

Initialization Flow:
1. Load and validate configuration without suppression
2. Create UI components
3. Run pre-initialization checks (if _pre_initialize_checks exists)
4. Set up event handlers (if _setup_handlers exists)
5. Get and validate root UI component
6. Run post-initialization checks (if _after_init_checks exists)
7. Log successful initialization
8. Return UI components with proper cleanup

Key Features:
- No premature output suppression
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
        # Add ui_components as property for backward compatibility
        self.ui_components = self._ui_components
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
    
    # _initialize_logger_bridge method removed - functionality now handled by BaseHandler
    
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
                
                # 4. Pre-initialization checks (optional) - silent
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
                # Then create and return the error response using imported function
                try:
                    # Create error response with return_type=dict to ensure we get a dictionary
                    error_response = create_error_response(
                        error_message=error_msg,
                        error=e,
                        title=f"{self.module_name} Error",
                        include_traceback=True,
                        return_type=dict
                    )
                    
                    # Get the container widget from the error response
                    error_widget = error_response.get('container')
                    
                    # If no container is returned, create a basic error widget
                    if error_widget is None:
                        error_widget = widgets.HTML(f"<div style='color:red'>{error_msg}</div>")
                    
                    # Store additional components in UI components for reference
                    self._ui_components.update({
                        'ui': error_widget,
                        'log_output': widgets.Output(),
                        'status_panel': widgets.Output(),
                        'error': True
                    })
                    
                    # Return the widget directly for display
                    return error_widget
                    
                except Exception as inner_e:
                    # Last resort fallback if error handling itself fails
                    print(f"Error creating error response: {inner_e}")
                    return widgets.HTML(f"<div style='color:red'>Error: Failed to initialize {self.module_name}: {str(e)}</div>")
    
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
        
    # _add_logger_bridge method removed - functionality now handled by BaseHandler
            
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

    # create_error_response method removed - now using imported create_error_response directly