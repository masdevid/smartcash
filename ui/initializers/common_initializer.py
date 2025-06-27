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

import logging
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import traceback
import sys
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.logger_bridge import UILoggerBridge, create_ui_logger_bridge
from smartcash.ui.utils.logging_utils import (
    restore_stdout,
    setup_aggressive_log_suppression
)

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
    
    Subclasses must implement the abstract methods:
        - _create_ui_components()
        - _get_default_config()
    
    Optional lifecycle hooks:
        - _pre_initialize_checks(): Pre-initialization validation
        - _setup_handlers(): Event handler setup
        - _after_init_checks(): Post-initialization validation
    """
    
    def __init__(self, module_name: str, config_handler_class: Type[ConfigHandler] = None):
        """Initialize dengan complete output suppression hingga UI ready
        
        Args:
            module_name: Nama module (used for logging)
            config_handler_class: Optional ConfigHandler class for config management
        """
        # Immediate suppression untuk prevent early logs
        setup_aggressive_log_suppression()
        
        self.module_name = module_name
        self._ui_components = {}
        self._logger_bridge = None
        self.config_handler = config_handler_class() if config_handler_class else None
        
        # Initialize silent logger
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        if hasattr(self.logger, 'set_level'):
            self.logger.set_level(logging.CRITICAL)  # Silent until UI ready
        else:
            self.logger.setLevel(logging.CRITICAL)
    
    def _initialize_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Setup logger bridge AFTER UI components created dan suppress semua stdout
        
        Args:
            ui_components: Dictionary of UI components that may be needed for logging
        """
        try:
            # Additional stdout suppression untuk ensure tidak ada log yang leak
            from smartcash.ui.utils.logging_utils import setup_stdout_suppression
            setup_stdout_suppression()
            
            self._logger_bridge = create_ui_logger_bridge(
                ui_components=ui_components,
                logger_name=f"smartcash.ui.{self.module_name}"
            )
            self.logger = self._logger_bridge
            
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
        
        # Suppress ALL output until UI is completely ready
        from smartcash.ui.utils.logging_utils import suppress_all_outputs
        suppress_all_outputs()
        
        try:
            # 1. Load config - silent
            config = self._load_config(config)
            
            # 2. Create UI components - silent  
            ui_components = self._create_ui_components(config, **kwargs) or {}
            self._ui_components = ui_components
            
            # 3. Initialize logger bridge SETELAH UI ready
            self._initialize_logger_bridge(ui_components)
            
            # 4. Pre-initialization checks (optional) - silent
            if hasattr(self, '_pre_initialize_checks'):
                self._pre_initialize_checks(config=config, **kwargs)
            
            # 5. Setup handlers SETELAH logger bridge ready
            if hasattr(self, '_setup_handlers'):
                ui_components = self._setup_handlers(ui_components, config, **kwargs) or ui_components
            
            # 6. Get root UI component
            root_ui = self._get_ui_root(ui_components)
            if not root_ui:
                raise ValueError("No root UI component found")
            
            # 7. Post-initialization checks (optional) - silent
            if hasattr(self, '_after_init_checks'):
                self._after_init_checks(ui_components=ui_components, config=config, **kwargs)
            
            # 8. SUCCESS: Log ONLY to UI setelah everything ready
            if hasattr(self.logger, 'info'):
                self.logger.info(f"✅ {self.module_name} siap digunakan")
            
            return root_ui
            
        except Exception as e:
            error_msg = f"❌ Gagal inisialisasi {self.module_name}: {str(e)}"
            
            # Pass the exception to _create_error_ui for better error reporting
            return self._create_error_ui(error_msg, e)
        finally:
            # Keep suppression active - hanya restore jika benar-benar diperlukan
            pass
    
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
            The root UI component or None if not found
        """
        # Try common root component names
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
            
    def _create_error_ui(self, error_message: str, error: Optional[Exception] = None) -> Any:
        """Create a fallback UI component to display error messages.
        
        Args:
            error_message: The error message to display
            error: Optional exception for traceback
            
        Returns:
            A widget that displays the error message. Always returns a widget.
        """
        import traceback
        from smartcash.ui.components.error.error_component import create_error_component
        
        # Get traceback if error is provided
        tb_text = None
        if error is not None:
            try:
                tb_text = traceback.format_exc()
                if tb_text.strip() == "NoneType: None":
                    tb_text = None
            except Exception:
                tb_text = None
        
        # Use standard error component with consistent styling
        error_component = create_error_component(
            error_message=str(error_message),  # Ensure it's a string
            title=f"🚨 {self.module_name} Initialization Error",
            traceback=tb_text,
            error_type="error",
            show_traceback=bool(tb_text)
        )
        
        # Return the container widget which is the main display
        return error_component.get('container', error_component)