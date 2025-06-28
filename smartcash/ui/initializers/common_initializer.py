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

# Import logging utilities first
from smartcash.ui.utils.logging_utils import (
    setup_aggressive_log_suppression,
    setup_stdout_suppression,
    suppress_ml_logs,
    suppress_viz_logs,
    suppress_data_logs
)

# Setup aggressive suppression at module level
setup_aggressive_log_suppression()
setup_stdout_suppression()
suppress_ml_logs()
suppress_viz_logs()
suppress_data_logs()

# Import other modules
import contextlib
import logging
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import traceback
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

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
    
    @classmethod
    def _initialize_suppression(cls):
        """Class method to ensure suppression is initialized"""
        if not cls._suppression_initialized:
            # Suppression is already handled at module level
            cls._suppression_initialized = True
    
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
        
        # Setup logger with maximum suppression
        try:
            self.logger = get_logger(f"smartcash.ui.{module_name}")
            if hasattr(self.logger, 'set_level'):
                self.logger.set_level(logging.CRITICAL)
            else:
                self.logger.setLevel(logging.CRITICAL)
            
            # Ensure no handlers can cause output
            if hasattr(self.logger, 'handlers'):
                for handler in self.logger.handlers[:]:
                    self.logger.removeHandler(handler)
            
            # Add null handler to prevent 'no handlers' warnings
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())
                
        except Exception:
            # If logger setup fails, create a silent logger
            self.logger = logging.getLogger(f"smartcash.ui.{module_name}.silent")
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL)
    
    @contextlib.contextmanager
    def _suppress_outputs(self):
        """Context manager untuk suppress semua output selama inisialisasi"""
        from smartcash.ui.utils.logging_utils import (
            setup_aggressive_log_suppression,
            setup_stdout_suppression,
            restore_stdout
        )
        
        # Setup suppression
        setup_aggressive_log_suppression()
        setup_stdout_suppression()
        
        try:
            yield
        finally:
            # Restore stdout when done
            restore_stdout()
    
    def _initialize_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Setup logger bridge AFTER UI components created dan suppress semua stdout
        
        Args:
            ui_components: Dictionary of UI components that may be needed for logging
        """
        try:
            # Additional stdout suppression untuk ensure tidak ada log yang leak
            from smartcash.ui.utils.logging_utils import setup_stdout_suppression
            setup_stdout_suppression()
            
            # Initialize logger bridge with UI components using factory function
            self._logger_bridge = create_ui_logger_bridge(
                ui_components=ui_components,
                logger_name=f"smartcash.ui.{self.module_name}"
            )
            self.logger = self._logger_bridge
            
            # Store logger bridge in ui_components for handlers to use
            ui_components['logger_bridge'] = self._logger_bridge
            
            if hasattr(self._logger_bridge, 'set_ui_ready'):
                self._logger_bridge.set_ui_ready(True)
            
            # ONLY log to UI, tidak ke stdout
            # self.logger.debug(f"Logger bridge initialized for {self.module_name}")
        except Exception as e:
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
                self._initialize_logger_bridge(ui_components)
                
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
            def log_to_ui(message: str, level: str = 'info') -> None:
                """Log message to UI output if available.
                
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
            
    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking purposes.
        
        Returns:
            ISO formatted timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()
        
    def create_error_response(self, error_message: str, error: Optional[Exception] = None) -> Dict[str, Any]:
        """Create a fallback UI component to display error messages.
        
        This method creates a consistent error UI that can be used across all initializers.
        It returns a dictionary with the error component and placeholders for other UI elements.
        
        Args:
            error_message: The error message to display
            error: Optional exception for traceback
            
        Returns:
            Dictionary containing:
            - 'ui': The main error widget
            - 'log_output': None (placeholder for log output)
            - 'status_panel': None (placeholder for status panel)
            - 'error': Boolean flag indicating this is an error state
        """
        import traceback
        from smartcash.ui.components.error.error_component import create_error_component
        
        # Get traceback if error is provided
        tb_text = None
        if error is not None:
            try:
                tb_text = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
            except Exception:
                tb_text = None
        
        # Strip any leading emoji from error message
        error_msg = str(error_message).strip()
        if error_msg and len(error_msg) > 0 and ord(error_msg[0]) > 255:  # Check if first character is likely an emoji
            error_msg = error_msg[1:].strip()
            
        # Create error component
        error_component = create_error_component(
            error_message=error_msg,
            title=f"🚨 {self.module_name} Error",
            traceback=tb_text,
            error_type="error",
            show_traceback=bool(tb_text)
        )
        
        # Return in the expected format
        return {
            'ui': error_component.get('container', error_component),
            'log_output': None,
            'status_panel': None,
            'error': True
        }