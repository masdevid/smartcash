"""
File: smartcash/ui/model/pretrained/pretrained_initializer.py
Description: Pretrained models initializer following ModuleInitializer pattern

Initialization Flow:
1. Load and validate configuration
2. Create UI components
3. Setup module handlers
4. Return UI with proper error handling
"""

import asyncio
from typing import Dict, Any, Optional, Type
from IPython.display import display

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.logger import get_module_logger
from .components.pretrained_ui import create_pretrained_ui
from .handlers.pretrained_ui_handler import PretrainedUIHandler
from .configs.pretrained_config_handler import PretrainedConfigHandler
from .services.pretrained_service import PretrainedService
from .constants import DEFAULT_CONFIG
from smartcash.ui.core.errors.handlers import create_error_response


class PretrainedInitializer(ModuleInitializer):
    """Pretrained models initializer with complete UI and backend service integration.
    
    Provides a structured approach to initializing the pretrained models module with
    proper error handling, logging, and UI component management.
    """
    
    def __init__(self):
        """Initialize pretrained module with configuration and services."""
        super().__init__(
            module_name='pretrained',
            config_handler_class=PretrainedConfigHandler,
            parent_module='model'
        )
        self.service = PretrainedService()
        self.ui_handler = None
    
    def create_ui_components(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Create pretrained models UI components.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating pretrained models UI components")
            
            # Get default config and merge with provided config
            final_config = DEFAULT_CONFIG.copy()
            if config:
                final_config.update(config)
            
            # Validate configuration
            validated_config = self.config_handler.validate_config(final_config)
            
            # Create UI components
            ui_result = create_pretrained_ui(validated_config, **kwargs)
            
            # Extract the actual UI components from the result
            ui_components = ui_result.get('ui_components', {})
            
            # Setup UI handler if not already set
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                self._ui_handler = PretrainedUIHandler(ui_components=ui_components)
                
                # Store service instance in the handler
                self._ui_handler.service = self.service
            
            # Update UI from config if components exist
            if ui_components:
                self.config_handler.update_ui_from_config(ui_components, validated_config)
                
                # Schedule post-init check
                self._schedule_post_init_check(ui_components, validated_config)
            else:
                self.logger.warning("⚠️ No UI components were created")
            
            self.logger.info(f"✅ Created {len(ui_components)} pretrained models UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create pretrained models UI components: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create pretrained models UI: {str(e)}") from e
    
    def _initialize_handlers(self, ui_components: Dict[str, Any], **kwargs) -> bool:
        """Initialize pretrained models UI handlers.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional initialization parameters
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.logger.info("🔧 Initializing pretrained models UI handlers")
            
            # Initialize UI handler if not already done
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                self._ui_handler = PretrainedUIHandler(ui_components=ui_components)
                
                # Store service instance in the handler
                self._ui_handler.service = self.service
            
            self.logger.info("✅ Pretrained models handlers initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pretrained models handlers: {e}", exc_info=True)
            return False
    
    def _schedule_post_init_check(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Schedule post-initialization checks.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._post_init_check(ui_components, config))
            else:
                loop.run_until_complete(self._post_init_check(ui_components, config))
        except Exception as e:
            self.logger.warning(f"Could not schedule post-init check: {str(e)}")
    
    async def _post_init_check(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Perform post-initialization checks for existing pretrained models.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            # Get UI handler
            ui_handler = self._ui_handler
            if not ui_handler:
                return
            
            # Check models status
            models_status = await ui_handler.check_models_status()
            
            # Update log with status
            log = ui_components.get('log_output')
            if log:
                for model, status in models_status.items():
                    if status['exists']:
                        log.append_stdout(f"✅ {model} is already downloaded\n")
                    else:
                        log.append_stdout(f"ℹ️ {model} will be downloaded when needed\n")
            
        except Exception as e:
            self.logger.error(f"❌ Error during post-init check: {e}", exc_info=True)
    
    def _initialize_impl(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Implementation of pretrained module initialization.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing UI components
        """
        try:
            # Create UI components
            ui_components = self.create_ui_components(config=config, **kwargs)
            
            # Initialize handlers
            if not self._initialize_handlers(ui_components, **kwargs):
                raise RuntimeError("Failed to initialize pretrained models handlers")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Failed to initialize pretrained module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return create_error_response(
                error_message=error_msg,
                error_type="PretrainedInitializationError",
                details=str(e),
                module_name=self.module_name
            )


# Global instances
_pretrained_initializer = PretrainedInitializer()


class PretrainedDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for pretrained models module"""
    
    def __init__(self):
        super().__init__(module_name="pretrained", parent_module="model")
        self._pretrained_initializer = PretrainedInitializer()
    
    def _initialize_impl(self, **kwargs):
        """Implementation using existing PretrainedInitializer"""
        return self._pretrained_initializer._initialize_impl(**kwargs)
        
    def display(self, **kwargs):
        """Display the pretrained models UI.
        
        Args:
            **kwargs: Additional arguments to pass to the initializer
        """
        try:
            # Get the UI components
            components = self._initialize_impl(**kwargs)
            
            # Display the main container if it exists
            if 'container' in components:
                from IPython.display import display as ipy_display
                ipy_display(components['container'])
                
            return components
            
        except Exception as e:
            error_msg = f"❌ Failed to display pretrained models UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Display error message
            from IPython.display import display as ipy_display, HTML
            ipy_display(HTML(f'<div class="alert alert-danger">{error_msg}</div>'))
            
            return {
                'status': 'error',
                'error': str(e),
                'message': error_msg
            }


# Global display initializer instance
_pretrained_display_initializer = PretrainedDisplayInitializer()


def get_pretrained_initializer() -> PretrainedInitializer:
    """Get the global pretrained initializer instance.
    
    Returns:
        PretrainedInitializer: The global pretrained initializer instance
    """
    global _pretrained_initializer
    if _pretrained_initializer is None:
        _pretrained_initializer = PretrainedInitializer()
    return _pretrained_initializer


def initialize_pretrained_ui(env=None, config=None, **kwargs):
    """Initialize and display pretrained models UI using DisplayInitializer
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Note:
        This function displays the UI directly and returns None.
        Use get_pretrained_components() if you need access to the components dictionary.
    """
    if env is not None:
        kwargs['env'] = env
    if config is not None:
        kwargs['config'] = config
    
    # Display the UI and return None
    _pretrained_display_initializer.display(**kwargs)


def get_pretrained_components(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Get pretrained models components dictionary without displaying UI
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components
    """
    if env is not None:
        kwargs['env'] = env
    if config is not None:
        kwargs['config'] = config
    
    return _pretrained_display_initializer.get_components(**kwargs)


def display_pretrained_ui(env=None, config=None, **kwargs):
    """Display pretrained models UI (alias for initialize_pretrained_ui)
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_pretrained_ui(env=env, config=config, **kwargs)


# Main entry point function for cell execution
def init_pretrained_ui(**kwargs):
    """Initialize and display pretrained models UI.
    
    This is the main entry point function that should be called from notebook cells.
    It creates the pretrained models initializer and displays the UI directly.
    
    Args:
        **kwargs: Additional initialization parameters
        
    Returns:
        Dictionary containing initialization results and UI components
    """
    try:
        # Get or create the global initializer
        initializer = get_pretrained_initializer()
        
        # Initialize and get UI components
        components = initializer.initialize_full(**kwargs)
        
        # Display the UI
        if 'container' in components:
            display(components['container'])
        
        # Log success
        initializer.logger.info("✅ Pretrained models UI initialized successfully")
        
        return {
            'status': 'success',
            'components': components,
            'initializer': initializer
        }
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize pretrained models UI: {str(e)}"
        get_module_logger('pretrained').error(error_msg, exc_info=True)
        
        # Display error message
        from IPython.display import display, HTML
        display(HTML(f'<div class="alert alert-danger">{error_msg}</div>'))
        
        return {
            'status': 'error',
            'error': str(e),
            'message': error_msg
        }


def _pretrained_initialize_legacy(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Legacy function wrapper for pretrained module initialization.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing UI components
    """
    initializer = PretrainedInitializer()
    return initializer._initialize_impl(config, **kwargs)