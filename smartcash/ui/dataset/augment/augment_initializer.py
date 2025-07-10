"""
File: smartcash/ui/dataset/augment/augment_initializer.py
Description: Main initializer for augment module following core patterns

This initializer inherits from core ModuleInitializer and implements augment-specific
initialization while preserving all original business logic.
"""

from typing import Dict, Any, List, Optional
import logging
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.core.errors.handlers import handle_ui_errors, create_error_response
from smartcash.ui.core.errors.decorators import handle_errors

from .components.augment_ui import create_augment_ui
from .handlers.augment_ui_handler import AugmentUIHandler
from .configs.augment_config_handler import AugmentConfigHandler
from .constants import UI_CONFIG


class AugmentInitializer(ModuleInitializer):
    """
    Main initializer for augment module following the standard pattern.
    
    Features:
    - 🏗️ Inherits from core ModuleInitializer
    - 🎨 Preserved original business logic
    - 🔄 Container-based UI architecture
    - ✅ Comprehensive error handling
    - 📊 Real-time progress tracking
    - 🗃️ Configuration management
    """
    
    def __init__(self):
        """Initialize the augment module."""
        super().__init__(
            module_name=UI_CONFIG['module_name'],
            parent_module=UI_CONFIG['parent_module'],
            config_handler_class=AugmentConfigHandler
        )
        self.ui_handler: Optional[AugmentUIHandler] = None
        self.logger = logging.getLogger(f"smartcash.ui.{UI_CONFIG['parent_module']}.{UI_CONFIG['module_name']}")
        self.logger.info(f"🎨 {UI_CONFIG['title']} initializer created")
        self._is_initialized = False  # Initialize the attribute explicitly
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Create augment UI components.
        
        Args:
            config: Configuration dictionary
            env: Optional environment context
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating augment UI components")
            
            # Create UI components
            ui_components = create_augment_ui(config=config, **kwargs)
            
            # Add module metadata
            ui_components.update({
                'module_name': UI_CONFIG['module_name'],
                'parent_module': UI_CONFIG['parent_module'],
                'ui_initialized': True,
                'config': config,
                'env': env,
                'module_version': UI_CONFIG['version'],
                'module_info': UI_CONFIG
            })
            
            # Ensure main container is set as 'ui' for DisplayInitializer
            if 'main_container' in ui_components and 'ui' not in ui_components:
                ui_components['ui'] = ui_components['main_container']
            
            self.logger.info(f"✅ UI components created: {len(ui_components)} components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create UI components: {e}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI augmentasi")
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup module handlers.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            env: Optional environment context
            **kwargs: Additional arguments
            
        Returns:
            Updated UI components with handlers
        """
        try:
            self.logger.info("🔧 Setting up augment UI handlers")
            
            # Create and setup UI handler
            self.ui_handler = AugmentUIHandler(ui_components)
            self.ui_handler.setup_handlers()
            
            # Store references
            ui_components.update({
                'initializer': self,
                'config_handler': self.config_handler,
                'ui_handler': self.ui_handler,
                'initialization_success': True
            })
            
            # Apply configuration to UI if available
            if config and self.config_handler:
                self.config_handler.update_ui_from_config(ui_components, config)
            
            self.logger.info("✅ Module handlers setup complete")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup module handlers: {e}", exc_info=True)
            return ui_components  # Return original components to avoid breaking the UI
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for augment module.
        
        Returns:
            Default configuration dictionary
        """
        try:
            from .configs.augment_defaults import get_default_augment_config
            default_config = get_default_augment_config()
            self.logger.debug("Default config loaded successfully")
            return default_config
        except Exception as e:
            self.logger.error(f"Failed to get default config: {e}", exc_info=True)
            # Return minimal working config
            return {
                'augment': {
                    'enabled': True,
                    'batch_size': 32,
                    'num_workers': 4
                }
            }
    
    def _get_critical_components(self) -> List[str]:
        """
        Get list of critical UI components that must exist.
        
        Returns:
            List of critical component keys
        """
        return [
            'ui', 'header_container', 'form_container', 'action_container',
            'operation_container', 'footer_container', 'progress_tracker'
        ]
    
    def pre_initialize_checks(self, **kwargs) -> None:
        """
        Perform pre-initialization checks.
        
        Raises:
            RuntimeError: If any check fails
        """
        try:
            import IPython
            # Add any additional checks here
        except ImportError:
            raise RuntimeError("Augment module requires IPython environment")
    
    def post_initialize_cleanup(self) -> None:
        """Perform any cleanup after initialization."""
        self.logger.debug("Post-initialization cleanup complete")
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implementation of the abstract _initialize_impl method.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        # Pre-initialization checks
        self.pre_initialize_checks(**kwargs)
        
        # Load configuration
        config = kwargs.get('config')
        if config is None:
            config = self._get_default_config()
        
        # Create UI components
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['config', 'env']}
        ui_components = self._create_ui_components(config, env=kwargs.get('env'), **filtered_kwargs)
        
        # Setup module handlers
        ui_components = self._setup_module_handlers(ui_components, config, env=kwargs.get('env'), **filtered_kwargs)
        
        # Post-initialization cleanup
        self.post_initialize_cleanup()
        
        return ui_components
    
    # Maintain backward compatibility
    def get_config_handler(self) -> Optional[AugmentConfigHandler]:
        """Get the configuration handler."""
        return self.config_handler
    
    def get_ui_handler(self) -> Optional[AugmentUIHandler]:
        """Get the UI handler."""
        return self.ui_handler
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update module configuration.
        
        Args:
            new_config: New configuration to apply
        """
        try:
            if self.config_handler:
                # Validate new configuration
                is_valid, errors = self.config_handler.validate_config(new_config)
                if not is_valid:
                    raise ValueError(f"Invalid configuration: {errors}")
                
                # Update configuration
                self.initial_config = new_config
                self.config_handler.update_config(new_config)
                self.logger.info("✅ Configuration updated successfully")
            else:
                self.logger.warning("⚠️ No config handler available for update")
        except Exception as e:
            self.logger.error(f"❌ Configuration update failed: {e}", exc_info=True)
            raise
    
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get current operation status.
        
        Returns:
            Dictionary containing operation status information
        """
        return {
            'module_initialized': True,
            'ui_handler_ready': self.ui_handler is not None,
            'config_handler_ready': self.config_handler is not None,
            'module_name': self.module_name,
            'parent_module': self.parent_module
        }


# Global instance and public API
_augment_initializer = AugmentInitializer()


class AugmentDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for augment module"""
    
    def __init__(self):
        super().__init__(
            module_name=UI_CONFIG['module_name'],
            parent_module=UI_CONFIG['parent_module']
        )
        self._augment_initializer = AugmentInitializer()
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of display initialization."""
        return self._augment_initializer.initialize(**kwargs)


# Global display initializer instance
_augment_display_initializer = AugmentDisplayInitializer()


def initialize_augment_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """
    Initialize and display augment UI using DisplayInitializer.
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of UI components
    """
    try:
        return _augment_initializer.initialize(env=env, config=config, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Failed to initialize augment UI: {e}")
        raise


def get_augment_components(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """
    Get augment components dictionary without displaying UI.
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of UI components
    """
    return initialize_augment_ui(env=env, config=config, display_ui=False, **kwargs)


def display_augment_ui(env=None, config=None, **kwargs) -> None:
    """
    Display augment UI (alias for initialize_augment_ui).
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_augment_ui(env=env, config=config, **kwargs)
def get_augment_config_handler(**kwargs) -> AugmentConfigHandler:
    """
    Factory function to create an augment config handler.
    
    Args:
        **kwargs: Arguments to pass to AugmentConfigHandler constructor
        
    Returns:
        AugmentConfigHandler instance
    """
    return AugmentConfigHandler(**kwargs)