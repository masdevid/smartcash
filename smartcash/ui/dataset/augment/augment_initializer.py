"""
File: smartcash/ui/dataset/augment/augment_initializer.py
Description: Main initializer for augment module following core patterns

This initializer inherits from core BaseInitializer and implements augment-specific
initialization while preserving all original business logic.
"""

from typing import Dict, Any, Optional
import logging
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.errors.decorators import handle_errors

from .components.augment_ui import create_augment_ui
from .handlers.augment_ui_handler import AugmentUIHandler
from .configs.augment_config_handler import AugmentConfigHandler
from .constants import UI_CONFIG


class AugmentInitializer(DisplayInitializer):
    """
    Main initializer for augment module with core inheritance patterns.
    
    Features:
    - 🏗️ Inherits from core ModuleInitializer
    - 🎨 Preserved original business logic
    - 🔄 Container-based UI architecture
    - ✅ Comprehensive error handling
    - 📊 Real-time progress tracking
    - 🗃️ Configuration management
    """
    
    @handle_ui_errors(error_component_title="Augment Initializer Error")
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize augment module.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(
            module_name=UI_CONFIG['module_name'],
            parent_module=UI_CONFIG['parent_module']
        )
        
        # Module-specific components
        self.ui_handler: Optional[AugmentUIHandler] = None
        self.config_handler: Optional[AugmentConfigHandler] = None
        self.initial_config = config or {}
        
        # Add logger for compatibility
        self.logger = logging.getLogger(f"smartcash.ui.{UI_CONFIG['parent_module']}.{UI_CONFIG['module_name']}")
        
        self.logger.info(f"🎨 {UI_CONFIG['title']} initializer created")
    
    @handle_errors(error_msg="Failed to initialize augment UI", reraise=True)
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implementation of augment module initialization.
        
        Args:
            **kwargs: Additional initialization arguments
            
        Returns:
            Dictionary containing initialized UI components for display
        """
        self.logger.info("🚀 Starting augment module initialization")
        
        try:
            # Get configuration from kwargs or use initial config
            config = kwargs.get('config', self.initial_config)
            
            # Step 1: Create configuration handler
            self.config_handler = AugmentConfigHandler(
                module_name=self.module_name,
                parent_module=self.parent_module,
                default_config=config
            )
            
            # Step 2: Create UI components
            ui_components = create_augment_ui(config)
            
            # Step 3: Create and setup UI handler
            self.ui_handler = AugmentUIHandler(ui_components)
            self.ui_handler.setup_handlers()
            
            # Step 4: Apply configuration to UI
            if config:
                self.config_handler.update_ui_from_config(ui_components, config)
            
            # Step 5: Add initialization metadata
            ui_components.update({
                'initializer': self,
                'config_handler': self.config_handler,
                'ui_handler': self.ui_handler,
                'initialization_success': True,
                'module_version': UI_CONFIG['version'],
                'module_info': UI_CONFIG
            })
            
            # Step 6: Add ui key for DisplayInitializer
            if 'main_container' in ui_components:
                ui_components['ui'] = ui_components['main_container']
            
            self.logger.info("✅ Augment module initialization completed successfully")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Augment module initialization failed: {e}")
            raise
    
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
            self.logger.error(f"❌ Configuration update failed: {e}")
            raise
    
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get current operation status.
        
        Returns:
            Dictionary containing operation status information
        """
        if self.ui_handler:
            return {
                'module_initialized': True,
                'ui_handler_ready': self.ui_handler is not None,
                'config_handler_ready': self.config_handler is not None,
                'module_name': self.module_name,
                'parent_module': self.parent_module
            }
        else:
            return {
                'module_initialized': False,
                'error': 'UI handler not available'
            }


# Factory function for creating augment UI (backward compatibility)
@handle_ui_errors(error_component_title="Augment UI Creation Error")
def initialize_augment_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Factory function to initialize augment UI.
    
    This function provides backward compatibility and a simple interface
    for creating augment UI components.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing initialized UI components
    """
    try:
        # Create initializer
        initializer = AugmentInitializer(config=config)
        
        # Initialize UI
        ui_components = initializer.initialize(**kwargs)
        
        return ui_components
        
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Failed to initialize augment UI: {e}")
        raise


# Factory function for config handler
def get_augment_config_handler(**kwargs) -> AugmentConfigHandler:
    """
    Factory function to create an augment config handler.
    
    Args:
        **kwargs: Arguments to pass to AugmentConfigHandler constructor
        
    Returns:
        AugmentConfigHandler instance
    """
    return AugmentConfigHandler(**kwargs)