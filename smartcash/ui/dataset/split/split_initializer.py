"""
File: smartcash/ui/dataset/split/split_initializer.py
Deskripsi: Split initializer yang mewarisi CommonInitializer dengan clean dependency

Initialization Flow:
1. Load and validate configuration
2. Create UI components
3. Setup module handlers
4. Return UI with proper error handling
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.dataset.split.components.split_ui import create_split_ui
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
from smartcash.ui.core.errors.handlers import create_error_response

# Constants
MODULE_NAME = "split_config"

class SplitInitializer(ModuleInitializer):
    """Split initializer dengan complete UI dan backend service integration

    Provides a structured approach to initializing the dataset split module with
    proper error handling, logging, and UI component management. Follows the same
    initialization flow as CommonInitializer with additional split-specific
    functionality.
    """

    def __init__(self):
        if hasattr(self, '_initialized'):
            self.logger.debug("SplitInitializer already initialized, reusing instance")
            return
            
        super().__init__(
            module_name=MODULE_NAME,
            config_handler_class=SplitConfigHandler
        )
        self.logger.debug("🔧 Initializing SplitInitializer")
        self.components = {}
        self._initialized = False
    
    def _initialize_impl(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Implementation of the initialization process.
        
        This method is called by the parent class's initialize() method.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments
                - display: If True, display the UI after initialization (default: True)
                - parent_display: Parent display to use for displaying the UI
            
        Returns:
            Dictionary containing UI components and other initialization data
        """
        try:
            # Ensure we have a config
            config = config or {}
            
            # Create UI components
            self.components = self._create_ui_components(config, **kwargs)
            
            # Set up module handlers
            self._setup_module_handlers(self.components, config, **kwargs)
            
            # Display the UI if requested
            if kwargs.get('display', True):
                self.display(parent_display=kwargs.get('parent_display'))
            
            # Mark as initialized
            self._is_initialized = True
            
            return self.components
            
        except Exception as e:
            self.handle_error(f"Failed to initialize SplitInitializer: {str(e)}", exc_info=True)
            return create_error_response("Gagal menginisialisasi modul split")
    
    def display(self, parent_display=None):
        """Display the UI components.
        
        Args:
            parent_display: Optional parent display to use for the UI
            
        Note:
            If parent_display is provided, the UI will be displayed in the parent display.
            Otherwise, it will be displayed in the default output.
            
        Returns:
            Dictionary of UI components if successful, None otherwise
        """
        if not hasattr(self, 'components') or not self.components:
            self.logger.warning("No components to display. Initialize first.")
            return None
            
        main_container = self.components.get('main_container')
        if not main_container:
            self.logger.error("Main container not found in components")
            return None
            
        try:
            # Import IPython display if available
            try:
                from IPython.display import display as ipy_display
                if parent_display:
                    parent_display.clear_output(wait=True)
                    with parent_display:
                        ipy_display(main_container)
                else:
                    ipy_display(main_container)
            except ImportError:
                # Fallback to standard print if IPython is not available
                print("Displaying UI components:")
                print(main_container)
                
            return self.components
            
        except Exception as e:
            self.logger.error(f"Error displaying UI: {str(e)}", exc_info=True)
            return None
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create split UI components following colab/dependency pattern

        Args:
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating split UI components")
            
            # Create UI components with the new structure
            ui_components = create_split_ui(config=config, **kwargs)
            
            # Add module-specific metadata
            ui_components.update({
                'split_initialized': True,
                'module_name': MODULE_NAME,
                'logger': self.logger,
                'config': config,
                'env': env
            })

            self.logger.info(f"✅ UI components created successfully: {len(ui_components)} components")
            return ui_components
            
        except Exception as e:
            self.handle_error(f"Failed to create UI components: {str(e)}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI split")
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers following colab/dependency pattern

        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Updated UI components with handlers
        """
        try:
            self.logger.info("🔧 Setting up split operation handlers")
            
            # Get buttons from components
            buttons = ui_components.get('buttons', {})
            
            # Set up save button handler if available
            save_button = buttons.get('save_button') or ui_components.get('save_button')
            if save_button and hasattr(save_button, 'on_click'):
                def on_save_clicked(_):
                    try:
                        self.save_config()
                        self.logger.info("✅ Konfigurasi berhasil disimpan")
                    except Exception as e:
                        self.logger.error(f"Gagal menyimpan konfigurasi: {str(e)}", exc_info=True)
                
                save_button.on_click(on_save_clicked)
            
            # Set up reset button handler if available
            reset_button = buttons.get('reset_button') or ui_components.get('reset_button')
            if reset_button and hasattr(reset_button, 'on_click'):
                def on_reset_clicked(_):
                    try:
                        self.reset_config()
                        self.logger.info("🔄 Konfigurasi telah direset ke nilai default")
                    except Exception as e:
                        self.logger.error(f"Gagal mereset konfigurasi: {str(e)}", exc_info=True)
                
                reset_button.on_click(on_reset_clicked)
            
            self.logger.info("✅ Operation handlers set up successfully")
            return ui_components
            
        except Exception as e:
            self.handle_error(f"Failed to set up module handlers: {str(e)}", exc_info=True)
            return ui_components
    
    def save_config(self) -> None:
        """Save the current configuration.
        
        This method should be implemented by subclasses to handle saving
        the current UI state to configuration.
        """
        self.logger.info("💾 Saving configuration...")
        try:
            # Get current values from UI components
            config = {}
            
            # Update config with values from form components
            form_components = getattr(self, 'components', {}).get('form_components', {})
            for name, widget in form_components.items():
                if hasattr(widget, 'value'):
                    config[name] = widget.value
            
            # Save using config handler
            if hasattr(self, 'config_handler') and self.config_handler:
                self.config_handler.save(config)
                self.logger.info("✅ Configuration saved successfully")
            else:
                self.logger.warning("No config handler available to save configuration")
                
        except Exception as e:
            self.handle_error(f"Failed to save configuration: {str(e)}", exc_info=True)
            raise
    
    def reset_config(self) -> None:
        """Reset the configuration to default values.
        
        This method should be implemented by subclasses to handle resetting
        the UI components to their default values.
        """
        self.logger.info("🔄 Resetting configuration to defaults...")
        try:
            # Get default config
            default_config = {}
            if hasattr(self, 'config_handler') and self.config_handler:
                default_config = self.config_handler.get_default_config()
            
            # Reset form components to default values
            form_components = getattr(self, 'components', {}).get('form_components', {})
            for name, widget in form_components.items():
                if hasattr(widget, 'value') and name in default_config:
                    widget.value = default_config[name]
            
            self.logger.info("✅ Configuration reset successfully")
            
        except Exception as e:
            self.handle_error(f"Failed to reset configuration: {str(e)}", exc_info=True)
            raise
    
    def _log_error(self, message: str) -> None:
        """Log an error message to the UI log output.
        
        Args:
            message: The error message to log
        """
        self._log_message(f"❌ {message}")
    
    def _log_success(self, message: str) -> None:
        """Log a success message to the UI log output.
        
        Args:
            message: The success message to log
        """
        self._log_message(f"✅ {message}")
    
    def _log_message(self, message: str) -> None:
        """Log a message to the UI log output.

        Args:
            message: The message to log
        """
        if hasattr(self, 'components') and 'log_output' in self.components:
            log_output = self.components['log_output']
            # If it's a mock or has context manager protocol, use it
            if hasattr(log_output, '__enter__') and hasattr(log_output, '__exit__'):
                with log_output:
                    print(message)
            # If it's a mock with a value attribute (like an Output widget)
            elif hasattr(log_output, 'value'):
                log_output.value = f"{log_output.value}\n{message}"
            # If it's a mock with an append_stdout method
            elif hasattr(log_output, 'append_stdout'):
                log_output.append_stdout(f"{message}\n")
            # Last resort, just print
            else:
                print(message)


# Global instance - lazy initialization
def get_split_initializer():
    if not hasattr(get_split_initializer, '_instance'):
        get_split_initializer._instance = SplitInitializer()
    return get_split_initializer._instance

_split_initializer = None  # Will be initialized on first access


class SplitDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for split module"""
    
    def __init__(self):
        super().__init__(module_name=MODULE_NAME, parent_module="dataset")
        self._split_initializer = get_split_initializer()
    
    def _initialize_impl(self, **kwargs):
        """Implementation using existing SplitInitializer"""
        return self._split_initializer.initialize(**kwargs)


# Global display initializer instance - lazy initialization
def get_display_initializer():
    if not hasattr(get_display_initializer, '_instance'):
        get_display_initializer._instance = SplitDisplayInitializer()
    return get_display_initializer._instance

_split_display_initializer = None  # Will be initialized on first access


def initialize_split_ui(env=None, config=None, **kwargs):
    """Initialize and display split UI using DisplayInitializer
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
            - parent_display: Optional parent display to use for the UI
    
    Note:
        This function displays the UI directly and returns None.
        Use get_split_components() if you need access to the components dictionary.
    """
    initializer = get_display_initializer()
    initializer.initialize(env=env, config=config, **kwargs)
    return None


def get_split_components(env=None, config=None, **kwargs):
    """Get split components dictionary without displaying UI
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    
    Returns:
        Dictionary of UI components
    """
    initializer = get_split_initializer()
    return initializer.initialize(env=env, config=config, display=False, **kwargs)


def display_split_ui(env=None, config=None, **kwargs):
    """Display split UI (alias for initialize_split_ui)
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
            - parent_display: Optional parent display to use for the UI
    """
    initialize_split_ui(env=env, config=config, **kwargs)


# Public API
export_functions = [
    'SplitInitializer',
    'initialize_split_ui',
    'get_split_components',
    'display_split_ui'
]

# No backward compatibility aliases - using new pattern only

__all__ = export_functions
