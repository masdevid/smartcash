"""
File: smartcash/ui/dataset/preprocess/preprocess_initializer.py
Description: Preprocessing initializer with modern UI container structure
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer, create_ui_display_function
from smartcash.ui.dataset.preprocess.constants import UI_CONFIG, MODULE_METADATA
from smartcash.ui.dataset.preprocess.configs.preprocess_config_handler import PreprocessConfigHandler
from smartcash.ui.dataset.preprocess.components.preprocess_ui import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocess.handlers.preprocess_ui_handler import PreprocessUIHandler


class PreprocessInitializer(DisplayInitializer):
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
            parent_module=UI_CONFIG['parent_module']
        )
        
        # Store module metadata
        self.module_metadata = MODULE_METADATA
        
        # Add logger for compatibility
        import logging
        self.logger = logging.getLogger(f"smartcash.ui.{UI_CONFIG['parent_module']}.{UI_CONFIG['module_name']}")
    
    
    def create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create UI components for preprocessing module.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        self.logger.info("🎯 Creating preprocessing UI components")
        
        # Create UI components using modern container structure
        ui_components = create_preprocessing_main_ui(config)
        
        # Add module metadata
        ui_components.update({
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'module_metadata': self.module_metadata,
            'data_dir': config.get('data', {}).get('dir', 'data')
        })
        
        self.logger.info(f"✅ Created {len(ui_components)} UI components")
        return ui_components
    
    def create_config_handler(self, **kwargs) -> PreprocessConfigHandler:
        """
        Create config handler for preprocessing module.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            PreprocessConfigHandler instance
        """
        self.logger.info("🎯 Creating preprocessing config handler")
        
        # Get persistence and sharing settings
        persistence_enabled = kwargs.get('persistence_enabled', True)
        enable_sharing = kwargs.get('enable_sharing', True)
        
        # Create config handler
        config_handler = PreprocessConfigHandler(
            module_name=self.module_name,
            parent_module=self.parent_module,
            persistence_enabled=persistence_enabled,
            enable_sharing=enable_sharing
        )
        
        self.logger.info("✅ Created preprocessing config handler")
        return config_handler
    
    def create_module_handler(self, ui_components: Dict[str, Any], **kwargs) -> PreprocessUIHandler:
        """
        Create module handler for preprocessing module.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional arguments
            
        Returns:
            PreprocessUIHandler instance
        """
        self.logger.info("🎯 Creating preprocessing module handler")
        
        # Create module handler
        module_handler = PreprocessUIHandler(
            ui_components=ui_components,
            config_handler=self.config_handler,
            module_name=self.module_name,
            parent_module=self.parent_module
        )
        
        self.logger.info("✅ Created preprocessing module handler")
        return module_handler
    
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


# ==================== DISPLAY FUNCTION ====================

# Create display function using DisplayInitializer pattern
initialize_preprocess_ui = create_ui_display_function(
    module_name=UI_CONFIG['module_name'],
    parent_module=UI_CONFIG['parent_module'],
)


# ==================== FACTORY FUNCTION ====================

def create_preprocessing_initializer() -> PreprocessInitializer:
    """
    Factory function to create preprocessing initializer.
    
    Returns:
        PreprocessInitializer instance
    """
    return PreprocessInitializer()


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