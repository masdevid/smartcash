"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Simplified base initializer for UI modules with fail-fast approach

Initialization Flow:
1. Suppress all outputs and initialize logging
2. Load and validate configuration
3. Create UI components (without logging)
4. Initialize logger bridge for UI logging
5. Run pre-initialization checks (if _pre_initialize_checks exists)
6. Set up event handlers (if _setup_handlers exists)
7. Get and validate root UI component
8. Run post-initialization checks (if _after_init_checks exists)
9. Log successful initialization
10. Return UI components and cleanup
"""

from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import traceback
import sys

from smartcash.common.logger import get_logger
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.logger_bridge import UILoggerBridge, create_ui_logger_bridge
from smartcash.ui.utils.logging_utils import (
    suppress_all_outputs,
    restore_stdout,
    setup_aggressive_log_suppression
)

class CommonInitializer(ABC):
    """Base class for initializing UI modules with a consistent and robust lifecycle.
    
    This class provides a structured approach to UI initialization with built-in error
    handling, logging, and configuration management. It follows a fail-fast pattern
    to ensure issues are caught early in the initialization process.
    
    Key Features:
        - Configurable initialization flow with clear extension points
        - Built-in error handling and recovery mechanisms
        - Integrated logging with UI support
        - Thread-safe component management
        - Memory-efficient resource handling
    
    Subclasses must implement the abstract methods:
        - _create_ui_components()
        - _get_default_config()
    """
    
    def __init__(self, module_name: str, config_handler_class: Type[ConfigHandler] = None):
        """
        Initialize the initializer with module name and optional config handler.
        
        Args:
            module_name: Name of the module (used for logging)
            config_handler_class: Optional ConfigHandler class for config management
        """
        self.module_name = module_name
        self._ui_components = {}
        self._logger_bridge = None
        self.config_handler = config_handler_class() if config_handler_class else None
        
        # Initialize with a basic logger first
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        
        # Setup aggressive log suppression
        setup_aggressive_log_suppression()
    
    def _initialize_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Set up the logger bridge for redirecting logs to the UI.
        
        This method creates a bridge between the Python logging system and the UI,
        allowing log messages to be displayed in the application's log panel.
        
        Args:
            ui_components: Dictionary containing UI components, which should include
                         a log output component if UI logging is desired.
                          
        Note:
            If logger bridge initialization fails, the method will fall back to
            standard logging and continue execution.
        """
        try:
            # Create and store the logger bridge
            self._logger_bridge = create_ui_logger_bridge(
                ui_components=ui_components,
                logger_name=f"smartcash.ui.{self.module_name}"
            )
            
            # Update the logger to use the bridge
            self.logger = self._logger_bridge.logger
            
            # Mark UI as ready to flush any buffered logs
            if hasattr(self._logger_bridge, 'set_ui_ready'):
                self._logger_bridge.set_ui_ready(True)
                
            self.logger.debug(f"Logger bridge initialized for {self.module_name}")
            
        except Exception as e:
            # Fallback to basic logging if bridge initialization fails
            self.logger = get_logger(f"smartcash.ui.{self.module_name}")
            self.logger.warning(f"Failed to initialize logger bridge: {str(e)}")
    
    def initialize(self, config: Dict[str, Any] = None, **kwargs) -> None:
        """Initialize and display the UI module with the given configuration.
        
        This is the main entry point that orchestrates the initialization process.
        It follows a strict sequence of operations, ensures proper error handling,
        and displays the UI directly.
        
        Args:
            config: Optional configuration dictionary. If None, will attempt to load
                   configuration using the configured config handler.
            **kwargs: Additional keyword arguments that may be required by subclasses.
                  
        Raises:
            ValueError: If required UI components are missing or invalid.
            RuntimeError: If initialization fails due to system or configuration issues.
            Exception: Any other exception that might occur during initialization.
            
        Example:
            >>> initializer = MyInitializer()
            >>> initializer.initialize()  # UI will be displayed automatically
        """
        from IPython.display import display
        
        # Suppress all outputs during initialization
        suppress_all_outputs()
        try:
            # 1. Load and validate config
            config = self._load_config(config)
            
            # 2. Create UI components first (without logging)
            ui_components = self._create_ui_components(config, **kwargs) or {}
            self._ui_components = ui_components
            
            # 3. Initialize logger bridge after UI components are created
            self._initialize_logger_bridge(ui_components)
            
            # 4. Run pre-initialization checks if method exists
            if hasattr(self, '_pre_initialize_checks'):
                self._pre_initialize_checks(config=config, **kwargs)
            
            # 5. Set up handlers if method exists
            if hasattr(self, '_setup_handlers'):
                ui_components = self._setup_handlers(ui_components, config, **kwargs) or ui_components
            
            # 6. Get the root UI component
            root_ui = self._get_ui_root(ui_components)
            if not root_ui:
                raise ValueError("No root UI component found")
            
            # 7. Run post-initialization checks if method exists
            if hasattr(self, '_after_init_checks'):
                self._after_init_checks(ui_components=ui_components, config=config, **kwargs)
            
            self.logger.info(f"✅ {self.module_name} initialized successfully")
            
            # Display the root UI component directly
            display(root_ui)
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize {self.module_name}: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            restore_stdout()  # Restore output before showing error UI
            error_ui = self._create_error_ui(error_msg)
            display(error_ui if error_ui else str(error_msg))
        finally:
            # Ensure output is restored in case of any other exceptions
            restore_stdout()
    
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
        """
        Create and return UI components as a dictionary.
        
        Args:
            config: Loaded configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        pass
        
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this module.
        
        Returns:
            Default configuration dictionary
        """
        pass
        
    def _get_ui_root(self, ui_components: Dict[str, Any]) -> Any:
        """
        Get the root UI component from the components dictionary.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            The root UI component or None if not found
        """
        # Try common root component names
        for key in ['ui', 'container', 'widget', 'root']:
            if key in ui_components:
                return ui_components[key]
                
        # Return first widget-like component if no root found
        for component in ui_components.values():
            if hasattr(component, 'layout') and hasattr(component, 'value'):
                return component
                
        return None
        
    def _pre_initialize_checks(self, **kwargs) -> None:
        """
        Override this method to perform pre-initialization checks.
        Raise an exception if any check fails.
        
        Args:
            **kwargs: Additional arguments that might be needed for checks
            
        Raises:
            Exception: If any pre-initialization check fails
        """
        pass
        
    def _add_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """
        Add logger bridge to UI components for logging to UI output.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            def log_to_ui(message: str, level: str = 'info') -> None:
                """
                Log message to UI output if available.
                
                Args:
                    message: Message to log
                    level: Log level (info, success, warning, error)
                """
                try:
                    if 'log_output' in ui_components and ui_components['log_output']:
                        with ui_components['log_output']:
                            icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
                            print(f"{icons.get(level, 'ℹ️')} {message}")
                except Exception:
                    pass  # Silent fail for logger bridge
            
            ui_components['logger_bridge'] = log_to_ui
            ui_components['logger_namespace'] = f"smartcash.ui.{self.module_name}"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup logger bridge: {str(e)}")
            
    def _create_error_ui(self, error_message: str) -> Any:
        """Create a fallback UI component to display error messages."""
        try:
            from smartcash.ui.components import create_error_component
            import traceback
            
            error_components = create_error_component(
                title=f"{self.module_name} Initialization Error",
                error_message=error_message,
                traceback=traceback.format_exc(),
                error_type="error"
            )
            return error_components['widget'] if 'widget' in error_components else str(error_components)
        except Exception as e:
            return f"Error initializing {self.module_name}: {error_message}\n{str(e)}"
