"""
File: smartcash/ui/model/train/training_initializer.py
Training module initializer following ModuleInitializer pattern.

Initialization Flow:
1. Load and validate configuration
2. Create UI components
3. Setup module handlers
4. Return UI with proper error handling
"""

import asyncio
from typing import Dict, Any, Optional

from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors.handlers import create_error_response

from .components.training_ui import create_training_ui
from .handlers.training_ui_handler import TrainingUIHandler
from .configs.training_config_handler import TrainingConfigHandler
from .constants import DEFAULT_CONFIG


class TrainingInitializer(ModuleInitializer):
    """Training module initializer with complete UI and backend service integration.
    
    Provides a structured approach to initializing the training module with
    proper error handling, logging, and UI component management.
    """
    
    def __init__(self):
        """Initialize training module with configuration and services."""
        super().__init__(
            module_name='train',
            config_handler_class=TrainingConfigHandler,
            parent_module='model'
        )
        self.service = None  # Will be initialized when needed
    
    def create_ui_components(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Create training UI components.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating training UI components")
            
            # Get default config and merge with provided config
            final_config = DEFAULT_CONFIG.copy()
            if config:
                final_config.update(config)
            
            # Validate configuration
            validated_config = self.config_handler.validate_config(final_config)
            
            # Create UI components
            ui_components = create_training_ui(validated_config, **kwargs)
            
            # Setup UI handler if not already set
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                self._ui_handler = TrainingUIHandler(ui_components=ui_components)
                
                # Setup event handlers with UI components
                self._ui_handler.setup(ui_components=ui_components)
            
            # Update UI from config
            self.config_handler.update_ui_from_config(ui_components, validated_config)
            
            # Schedule post-init check
            self._schedule_post_init_check(ui_components, validated_config)
            
            self.logger.info(f"✅ Created {len(ui_components)} training UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create training UI components: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create training UI: {str(e)}") from e
    
    def _initialize_handlers(self, ui_components: Dict[str, Any], **kwargs) -> bool:
        """Initialize training UI handlers.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional initialization parameters
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.logger.info("🔧 Initializing training UI handlers")
            
            # Initialize UI handler if not already done
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                self._ui_handler = TrainingUIHandler(ui_components=ui_components)
                
                # Setup event handlers
                self._ui_handler.setup(ui_components=ui_components)
            
            self.logger.info("✅ Training handlers initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize training handlers: {e}", exc_info=True)
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
        """Perform post-initialization checks.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            # Get UI handler
            ui_handler = self._ui_handler
            if not ui_handler:
                return
            
            # Check if training is already in progress
            # This is a placeholder for actual training status check
            if hasattr(ui_handler, 'check_training_status'):
                training_status = await ui_handler.check_training_status()
                log = ui_components.get('log_output')
                if log and training_status and training_status.get('is_running', False):
                    log.append_stdout("ℹ️ Training is already in progress\n")
        
        except Exception as e:
            self.logger.error(f"❌ Error during post-init check: {e}", exc_info=True)
    
    def _initialize_impl(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Implementation of training module initialization.
        
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
                raise RuntimeError("Failed to initialize training handlers")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Failed to initialize training module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return create_error_response(
                error_message=error_msg,
                error_type="TrainingInitializationError",
                details=str(e),
                module_name=self.module_name
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _post_init_setup(self, ui_components: Dict[str, Any]) -> None:
        """
        Perform post-initialization setup.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Validate backend availability
            if 'ui_handler' in ui_components:
                ui_handler = ui_components['ui_handler']
                service = ui_handler.training_service
                
                # Check backend availability
                backend_status = service.validate_backend_availability()
                
                if backend_status.get("available", False):
                    self.logger.info("✅ Backend training components available")
                else:
                    self.logger.warning(f"⚠️ Backend not available: {backend_status.get('message', 'Unknown')}")
                    self.logger.info("🔄 Running in simulation mode")
            
            # Set initial button states
            if 'ui_handler' in ui_components:
                ui_handler = ui_components['ui_handler']
                ui_handler._update_button_states()
            
            # Update initial config summary
            if 'ui_handler' in ui_components:
                ui_handler = ui_components['ui_handler']
                ui_handler._update_config_summary()
            
        except Exception as e:
            self.logger.warning(f"Post-initialization setup warning: {str(e)}")


# Global instances
_training_initializer = TrainingInitializer()


class TrainingDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for training module"""
    
    def __init__(self):
        super().__init__(module_name="train", parent_module="model")
        self._training_initializer = TrainingInitializer()
    
    def _initialize_impl(self, **kwargs):
        """Implementation using existing TrainingInitializer"""
        return self._training_initializer._initialize_impl(**kwargs)
        
    def display(self, **kwargs):
        """Display the training UI.
        
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
            error_msg = f"❌ Failed to display training UI: {str(e)}"
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
_training_display_initializer = TrainingDisplayInitializer()


def get_training_initializer() -> TrainingInitializer:
    """Get the global training initializer instance.
    
    Returns:
        TrainingInitializer: The global training initializer instance
    """
    global _training_initializer
    if _training_initializer is None:
        _training_initializer = TrainingInitializer()
    return _training_initializer


def initialize_training_ui(env=None, config=None, **kwargs):
    """Initialize and display training UI using DisplayInitializer
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Note:
        This function displays the UI directly and returns None.
        Use get_training_components() if you need access to the components dictionary.
    """
    if env is not None:
        kwargs['env'] = env
    if config is not None:
        kwargs['config'] = config
    
    # Display the UI and return None
    _training_display_initializer.display(**kwargs)


def get_training_components(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Get training components dictionary without displaying UI
    
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
    
    return _training_display_initializer.get_components(**kwargs)


def display_training_ui(env=None, config=None, **kwargs):
    """Display training UI (alias for initialize_training_ui)
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_training_ui(env=env, config=config, **kwargs)


# Main entry point function for cell execution
def init_training_ui(**kwargs):
    """Initialize and display training UI.
    
    This is the main entry point function that should be called from notebook cells.
    It creates the training initializer and displays the UI directly.
    
    Args:
        **kwargs: Additional initialization parameters
        
    Returns:
        Dictionary containing initialization results and UI components
    """
    try:
        # Get or create the global initializer
        initializer = get_training_initializer()
        
        # Initialize and get UI components
        components = initializer.initialize_full(**kwargs)
        
        # Display the UI
        if 'container' in components:
            from IPython.display import display
            display(components['container'])
        
        # Log success
        initializer.logger.info("✅ Training UI initialized successfully")
        
        return {
            'status': 'success',
            'components': components,
            'initializer': initializer
        }
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize training UI: {str(e)}"
        get_module_logger('training').error(error_msg, exc_info=True)
        
        # Display error message
        from IPython.display import display, HTML
        display(HTML(f'<div class="alert alert-danger">{error_msg}</div>'))
        
        return {
            'status': 'error',
            'error': str(e),
            'message': error_msg
        }