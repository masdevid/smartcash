"""
Dataset Split Initializer Module.

This module provides the main initialization code for the dataset split configuration UI,
following the container-based pattern used throughout the SmartCash application.
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display
import logging

# Core imports
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.core.errors.handlers import handle_ui_errors, create_error_response

# Local imports
from .components.split_ui import create_split_ui_components
from .handlers.config_handler import SplitConfigHandler

# Constants
MODULE_NAME = "split_config"

class SplitInitializer(DisplayInitializer):
    """Dataset split configuration UI implementation following container-based pattern.
    
    This class implements the dataset split configuration UI using the container-based
    pattern, providing a consistent user experience with other SmartCash modules.
    
    Features:
    - Container-based UI following SmartCash design patterns
    - Centralized configuration management
    - Consistent error handling and logging
    - Support for both interactive and programmatic usage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the dataset split configuration UI.
        
        Args:
            config: Optional configuration dictionary for split settings
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(module_name=MODULE_NAME, parent_module='dataset.split')
        self.config = config or {}
        self.kwargs = kwargs
        self.components: Dict[str, Any] = {}
        self.config_handler = SplitConfigHandler(config)
        
        self.logger.debug(f"Initialized with config: {bool(config)}")
            
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of the initialization process.
        
        This method is called by the parent class's initialize() method.
        
        Returns:
            Dict containing initialization status
        """
        try:
            # Create UI components
            self._create_ui_components(self.config or {})
            
            # Mark as initialized
            self._is_initialized = True
            
            return {
                'status': 'success',
                'message': 'SplitInitializer initialized successfully',
                'components': self.components
            }
            
        except Exception as e:
            error_msg = f"Failed to initialize SplitInitializer: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': 'error',
                'message': error_msg,
                'error': str(e)
            }
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create and return UI components using the container-based pattern.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing all UI components
        """
        try:
            # Create UI components using container pattern
            ui_components = create_split_ui_components(config, **self.kwargs)
            
            # Store references to components
            self.components = {
                **ui_components,
                'main_container': ui_components['main_container'],
                'form_components': ui_components.get('form_components', {})
            }
            
            # Set up event handlers
            self._setup_handlers()
            
            return self.components
            
        except Exception as e:
            error_msg = f"Gagal membuat komponen UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def display(self) -> None:
        """Display the UI using the container-based pattern."""
        if not hasattr(self, 'components') or 'main_container' not in self.components:
            self._create_ui_components(self.config, **self.kwargs)
            
        if 'main_container' in self.components:
            from IPython.display import display
            display(self.components['main_container'])
        else:
            self.logger.error("Main container not found in components")
    
    def _setup_handlers(self) -> None:
        """Set up event handlers for UI components.
        
        This method connects UI components to their respective handlers,
        including save/reset buttons and any custom event handlers.
        """
        if not hasattr(self, 'components') or not self.components:
            return
            
        form_components = self.components.get('form_components', {})
        
        # Set up save button handler
        save_button = form_components.get('save_button')
        if save_button and hasattr(self, 'save_config'):
            def on_save_clicked(button):
                try:
                    self.save_config()
                    self._log_success("Konfigurasi berhasil disimpan")
                except Exception as e:
                    self._log_error(f"Gagal menyimpan konfigurasi: {str(e)}")
            
            save_button.on_click(on_save_clicked)
        
        # Set up reset button handler
        reset_button = form_components.get('reset_button')
        if reset_button and hasattr(self, 'reset_ui'):
            def on_reset_clicked(button):
                try:
                    self.reset_ui()
                    self._log_success("UI berhasil direset ke nilai default")
                except Exception as e:
                    self._log_error(f"Gagal me-reset UI: {str(e)}")
            
            reset_button.on_click(on_reset_clicked)
    
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

@handle_ui_errors(error_component_title="Split Config Error", log_error=True)
def create_split_config_cell(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Create and display dataset split configuration UI.
    
    This is the main entry point for users to create and display the dataset
    split configuration UI in a notebook cell.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments passed to the initializer
    """
    try:
        initializer = SplitInitializer(config=config, **kwargs)
        initializer.initialize()
        initializer.display()
    except Exception as e:
        handle_ui_errors(e, "Gagal menginisialisasi UI konfigurasi split dataset")


@handle_ui_errors(error_component_title="Split Config Error", log_error=True)
def get_split_config_components(
    config: Optional[Dict[str, Any]] = None, 
    **kwargs
) -> Dict[str, Any]:
    """Create split config UI and return components for programmatic access.
    
    This function creates the dataset split UI components and returns them
    for programmatic access, without displaying the UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments passed to the initializer
               
    Returns:
        Dictionary containing all UI components and the initializer instance
    """
    try:
        initializer = SplitInitializer(config=config, **kwargs)
        initializer.initialize()
        
        # Ensure UI components are created
        if not hasattr(initializer, 'components') or not initializer.components:
            initializer._create_ui_components(initializer.config, **kwargs)
            
        return {
            'initializer': initializer,
            'components': initializer.components,
            'main_container': initializer.components.get('main_container'),
            'form_components': initializer.components.get('form_components', {})
        }
        
    except Exception as e:
        error_msg = f"Gagal mendapatkan komponen UI split config: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return create_error_response(
            error_type="UI Initialization Error",
            message=error_msg,
            details=str(e)
        )


# Standard entry point functions following UI module pattern

def init_split_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Initialize and display the split configuration UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments passed to the initializer
    """
    create_split_config_cell(config, **kwargs)


def get_split_initializer(config: Optional[Dict[str, Any]] = None, **kwargs) -> SplitInitializer:
    """Get a split initializer instance.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments passed to the initializer
        
    Returns:
        SplitInitializer instance
    """
    return SplitInitializer(config=config, **kwargs)
