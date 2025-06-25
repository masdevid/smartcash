"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Simplified base initializer for UI modules with fail-fast approach
"""

from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import traceback

from smartcash.common.logger import get_logger
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.logging_utils import suppress_all_outputs

class CommonInitializer(ABC):
    """
    Simplified base initializer for UI modules with fail-fast approach.
    
    Features:
    - Simple initialization flow
    - Fail-fast error handling
    - Minimal required methods
    - Clear separation of concerns
    """
    
    def __init__(self, module_name: str, config_handler_class: Type[ConfigHandler] = None):
        """
        Initialize the initializer with module name and optional config handler.
        
        Args:
            module_name: Name of the module (used for logging)
            config_handler_class: Optional ConfigHandler class for config management
        """
        self.module_name = module_name
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        self.config_handler = config_handler_class() if config_handler_class else None
        self.logger.debug(f"Initializing {module_name} module")
    
    def initialize(self, config: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Main initialization method that follows a simple fail-fast approach.
        
        Args:
            config: Optional config dict. If not provided, will try to load from config handler.
            **kwargs: Additional arguments passed to component creation
            
        Returns:
            The root UI component or error UI if initialization fails
        """
        try:
            suppress_all_outputs()
            self.logger.info(f"🚀 Starting {self.module_name} initialization")
            
            # 1. Load and validate config
            config = self._load_config(config)
            if not config:
                raise ValueError("Failed to load configuration")
                
            # 2. Create UI components (including log accordion)
            ui_components = self._create_ui_components(config, **kwargs)
            if not ui_components:
                raise ValueError("Failed to create UI components")
                
            # 3. Add logger bridge
            self._add_logger_bridge(ui_components)
                
            # 4. Setup handlers if implemented
            if hasattr(self, '_setup_handlers'):
                ui_components = self._setup_handlers(ui_components, config, **kwargs) or ui_components
                
            # 5. Get the root UI component
            root_ui = self._get_ui_root(ui_components)
            if not root_ui:
                raise ValueError("No root UI component found")
            
            # 6. Run pre-initialization checks (after everything is set up)
            try:
                self._pre_initialize_checks(**kwargs)
                self.logger.info(f"✅ {self.module_name} initialized successfully")
                return root_ui
            except Exception as e:
                error_msg = f"Initialization failed: {str(e)}"
                self.logger.error(error_msg)
                return self._create_error_ui(error_msg)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {self.module_name}: {str(e)}\n{traceback.format_exc()}")
            return self._create_error_ui(str(e))
    
    def _load_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load and validate configuration"""
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
        """
        Create an error UI component using the shared ErrorComponent.
        
        Args:
            error_message: Error message to display
            
        Returns:
            Error UI component
        """
        try:
            from smartcash.ui.components.error.error_component import create_error_component
            import traceback
            
            error_components = create_error_component(
                error_message=error_message,
                traceback=traceback.format_exc(),
                title=f"❌ {self.module_name} Initialization Error",
                error_type="error",
                show_traceback=True,
                width='100%'
            )
            return error_components['widget']
            
        except Exception as e:
            # Fallback to simple error message if component creation fails
            return f"Error initializing {self.module_name}: {error_message}\n{str(e)}"
