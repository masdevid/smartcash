"""
File: smartcash/ui/dataset/preprocess/preprocess_initializer.py
Description: Preprocessing initializer with modern UI container structure
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.dataset.preprocess.constants import UI_CONFIG, MODULE_METADATA
from smartcash.ui.dataset.preprocess.configs.preprocess_config_handler import PreprocessConfigHandler
from smartcash.ui.dataset.preprocess.components.preprocess_ui import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocess.handlers.preprocess_ui_handler import PreprocessUIHandler


class PreprocessInitializer(ModuleInitializer):
    """
    Preprocessing initializer with modern UI container structure.
    
    Features:
    - 🎯 Modern container-based UI architecture
    - 📊 Enhanced UI-Config synchronization
    - 🔧 Standardized handler management
    - 🔄 Consistent event registration
    - 📱 DisplayInitializer integration
    """
    
    def __init__(self):
        """Initialize preprocessing initializer."""
        super().__init__(
            module_name=UI_CONFIG['module_name'],
            config_handler_class=PreprocessConfigHandler
        )
        
        # Store module metadata
        self.module_metadata = MODULE_METADATA
        
        # Initialize handlers
        self._module_handler = None
    
    
    def create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create preprocessing UI components.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing UI components and module state
        """
        try:
            self.logger.info("🎯 Starting preprocessing module initialization")
            
            # Filter out any kwargs that create_preprocessing_main_ui doesn't expect
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['env']}
            
            # Create main UI components
            self.logger.info("🎯 Creating preprocessing UI components")
            ui_components = create_preprocessing_main_ui(config=config, **filtered_kwargs)
            
            if not ui_components:
                raise ValueError("Failed to create UI components: create_preprocessing_main_ui returned None")
                
            # Add module metadata to UI components
            ui_components['module_metadata'] = self.module_metadata
            self.logger.info("✅ Created UI components")
            
            # Create module handler
            self._module_handler = self.create_module_handler(ui_components, **kwargs)
            if not self._module_handler:
                raise ValueError("Failed to create module handler")
            self.logger.info("✅ Created module handler")
            
            # Prepare result dictionary
            result = {
                'ui_components': ui_components,
                'module_handler': self._module_handler,
                'ui': ui_components.get('ui')  # For backward compatibility
            }
            
            # Add UI components to top level for backward compatibility
            if isinstance(ui_components, dict):
                result.update(ui_components)
                
            self.logger.info("✅ Preprocessing module initialization complete")
            return result
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize preprocessing module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _connect_handlers(self, module_handler: 'PreprocessUIHandler', ui_components: Dict[str, Any]) -> None:
        """
        Connect UI components with their respective handlers.
        
        Args:
            module_handler: The module handler instance
            ui_components: Dictionary of UI components
        """
        try:
            # Get UI components
            button_run = ui_components.get('button_run')
            button_reset = ui_components.get('button_reset')
            
            # Connect button click handlers
            if button_run and hasattr(module_handler, 'on_run_clicked'):
                button_run.on_click(module_handler.on_run_clicked)
                
            if button_reset and hasattr(module_handler, 'on_reset_clicked'):
                button_reset.on_click(module_handler.on_reset_clicked)
                
            # Connect other UI components as needed
            # Example:
            # if 'some_dropdown' in ui_components and hasattr(module_handler, 'on_dropdown_change'):
            #     ui_components['some_dropdown'].observe(module_handler.on_dropdown_change, 'value')
                
            self.logger.info("✅ Connected UI event handlers")
            
        except Exception as e:
            error_msg = f"❌ Failed to connect UI event handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    def create_module_handler(self, ui_components: Dict[str, Any], **kwargs) -> PreprocessUIHandler:
        """
        Create module handler for preprocessing module.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional arguments (filtered to only include expected args)
            
        Returns:
            PreprocessUIHandler instance
        """
        try:
            # Filter kwargs to only include expected parameters
            expected_kwargs = {
                'module_name': self.module_name,
                'parent_module': self.parent_module
            }
            
            # Create our specific handler class directly
            module_handler = PreprocessUIHandler(
                ui_components=ui_components,
                config_handler=self.config_handler,
                **expected_kwargs
            )
            
            # Connect event handlers
            self._connect_handlers(module_handler, ui_components)
            
            self.logger.info("✅ Created preprocessing module handler")
            return module_handler
            
        except Exception as e:
            error_msg = f"❌ Failed to create preprocessing module handler: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def setup_handlers(self, ui_components: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Setup handlers for preprocessing module.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional arguments
            
        Returns:
            Updated UI components with handlers
        """
        self.logger.info("🎯 Setting up preprocessing handlers")
        
        # Add config handler to UI components
        ui_components['config_handler'] = self.config_handler
        
        # Create module handler
        module_handler = self.create_module_handler(ui_components)
        
        # Add module handler to UI components
        ui_components['module_handler'] = module_handler
        
        # Setup event handlers
        self._setup_event_handlers(ui_components, module_handler)
        
        self.logger.info("✅ Preprocessing handlers setup complete")
        return ui_components
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implementation of the abstract _initialize_impl method.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components for display
        """
        try:
            # Get configuration
            config = kwargs.get('config', {})
            
            # Create UI components
            # Remove config from kwargs to avoid conflict
            ui_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
            ui_components = self.create_ui_components(config, **ui_kwargs)
            
            # Create config handler if needed
            if not hasattr(self, 'config_handler'):
                self.config_handler = self.create_config_handler(**kwargs)
                
            # Load configuration (for potential future use)
            # loaded_config = self.config_handler.get_config()
            
            # Setup handlers
            ui_components = self.setup_handlers(ui_components, **kwargs)
            
            # Add ui key for DisplayInitializer
            if 'main_container' in ui_components:
                ui_components['ui'] = ui_components['main_container']
            
            # Return UI components directly for DisplayInitializer
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize preprocessing UI: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Failed to initialize preprocessing UI: {str(e)}",
                'error': str(e)
            }
    
    def _setup_event_handlers(self, ui_components: Dict[str, Any], module_handler: PreprocessUIHandler) -> None:
        """
        Setup event handlers for UI components.
        
        Args:
            ui_components: Dictionary of UI components
            module_handler: Module handler instance
        """
        # Setup button click handlers
        if 'preprocess_btn' in ui_components and ui_components['preprocess_btn'] is not None:
            ui_components['preprocess_btn'].on_click(
                lambda _: module_handler.handle_preprocess_click()
            )
        
        if 'check_btn' in ui_components and ui_components['check_btn'] is not None:
            ui_components['check_btn'].on_click(
                lambda _: module_handler.handle_check_click()
            )
        
        if 'cleanup_btn' in ui_components and ui_components['cleanup_btn'] is not None:
            ui_components['cleanup_btn'].on_click(
                lambda _: module_handler.handle_cleanup_click()
            )
        
        # Setup configuration change handlers
        if hasattr(module_handler, 'setup_config_handlers'):
            module_handler.setup_config_handlers(ui_components)
    
    def get_critical_components(self) -> List[str]:
        """
        Get list of critical UI components that must exist.
        
        Returns:
            List of critical component keys
        """
        return [
            'ui', 'main_container', 'preprocess_btn', 'check_btn', 'cleanup_btn',
            'operation_container', 'progress_tracker', 'log_accordion', 
            'header_container', 'form_container', 'action_container', 'footer_container'
        ]
    
    def pre_initialize_checks(self, **kwargs) -> None:
        """
        Perform pre-initialization checks.
        
        Raises:
            Exception: If any pre-initialization check fails
        """
        # Check if we're in a supported environment
        try:
            import IPython  # noqa: F401
            # Additional checks can be added here
        except ImportError:
            raise RuntimeError("Dataset preprocessing requires IPython environment")
        
        # Check if required backend modules are available
        try:
            import smartcash.dataset.preprocessor  # noqa: F401
        except ImportError:
            raise RuntimeError("Backend preprocessing module not available")


# ==================== FACTORY FUNCTION ====================

def create_preprocessing_initializer() -> PreprocessInitializer:
    """
    Factory function to create preprocessing initializer.
    
    Returns:
        PreprocessInitializer instance
    """
    return PreprocessInitializer()


# ==================== DISPLAY INITIALIZER ====================

class PreprocessDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for preprocessing module"""
    
    def __init__(self):
        super().__init__(module_name=UI_CONFIG['module_name'], 
                         parent_module=UI_CONFIG['parent_module'])
        self._preprocess_initializer = PreprocessInitializer()
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation using existing PreprocessInitializer"""
        return self._preprocess_initializer.initialize(**kwargs)


# Global display initializer instance
_preprocess_display_initializer = PreprocessDisplayInitializer()


# ==================== GLOBAL INSTANCE ====================

# Global instance for backward compatibility
_preprocessing_initializer = PreprocessInitializer()


def get_preprocessing_initializer() -> PreprocessInitializer:
    """
    Get the global preprocessing initializer instance.
    
    Returns:
        PreprocessInitializer instance
    """
    return _preprocessing_initializer


def initialize_preprocess_ui(env=None, config=None, **kwargs) -> None:
    """
    Initialize and display preprocessing UI using DisplayInitializer

    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    
    Note:
        This function displays the UI directly and returns None.
        Use get_preprocessing_initializer() if you need access to the components dictionary.
    """
    _preprocess_display_initializer.initialize_and_display(config=config, env=env, **kwargs)