"""
File: smartcash/ui/dataset/split/split_uimodule.py
Description: Main UIModule implementation for split configuration module
"""

from typing import Dict, Any, Optional

# Import ipywidgets for UI components
import ipywidgets as widgets

from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Import split components
from smartcash.ui.dataset.split.components.split_ui import create_split_ui
from smartcash.ui.dataset.split.configs.split_config_handler import SplitConfigHandler
from smartcash.ui.dataset.split.configs.split_defaults import get_default_split_config
from smartcash.ui.dataset.split.constants import UI_CONFIG, MODULE_METADATA


class SplitUIModule(BaseUIModule):
    """
    UIModule implementation for dataset split configuration.
    
    Features:
    - 📋 Split configuration management (train/val/test ratios)
    - 💾 Save and reset configuration functionality  
    - 🎯 New UIModule pattern with operation container logging
    - 🔧 Configuration validation and UI synchronization
    - 📱 Button bindings for save/reset operations
    - ♻️ Proper resource management and cleanup
    """
    
    def __init__(self):
        """Initialize split UI module."""
        super().__init__(
            module_name='split',
            parent_module='dataset'
        )
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'action_container',
            'operation_container'
        ]
        
        self.logger.debug("✅ SplitUIModule initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for split module."""
        return get_default_split_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> SplitConfigHandler:
        """Create config handler instance for split module."""
        return SplitConfigHandler(config)
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for split module."""
        try:
            from .components.split_ui import create_split_ui
            
            self.logger.debug("Creating split UI components...")
            ui_components = create_split_ui(config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def _register_default_operations(self) -> None:
        """Register default operations for split module."""
        # Call parent method first
        super()._register_default_operations()
        
        # Register split-specific operations
        self.register_operation_handler('get_split_status', self.get_split_status)
        self.register_operation_handler('split_dataset', self._handle_split_dataset)
        
        # Register custom button handlers
        self.register_button_handler('save', self._handle_save_config)
        self.register_button_handler('reset', self._handle_reset_config)
        self.register_button_handler('split_dataset', self._handle_split_dataset)
    
    def _handle_save_config(self, button=None):
        """Handle save config button click."""
        try:
            self.update_operation_status("Saving configuration...", "info")
            self.log("💾 Save config button clicked", 'info')
            
            result = self.save_config()
            if result.get('success'):
                success_msg = f"Configuration saved: {result.get('message', '')}"
                self.update_operation_status(success_msg, "info")
                self.log(f"✅ {success_msg}", 'info')
            else:
                error_msg = f"Save failed: {result.get('message', '')}"
                self.update_operation_status(error_msg, "error")
                self.log(f"❌ {error_msg}", 'error')
        except Exception as e:
            error_msg = f"Save config error: {e}"
            self.update_operation_status(error_msg, "error")
            self.log(f"❌ {error_msg}", 'error')
    
    def _handle_reset_config(self, button=None):
        """Handle reset config button click."""
        try:
            self.update_operation_status("Resetting configuration...", "info")
            self.log("🔄 Reset config button clicked", 'info')
            
            result = self.reset_config()
            if result.get('success'):
                success_msg = f"Configuration reset: {result.get('message', '')}"
                self.update_operation_status(success_msg, "info")
                self.log(f"✅ {success_msg}", 'info')
            else:
                error_msg = f"Reset failed: {result.get('message', '')}"
                self.update_operation_status(error_msg, "error")
                self.log(f"❌ {error_msg}", 'error')
        except Exception as e:
            error_msg = f"Reset config error: {e}"
            self.update_operation_status(error_msg, "error")
            self.log(f"❌ {error_msg}", 'error')
    
    def _handle_split_dataset(self, button=None):
        """Handle split dataset button click."""
        try:
            self.update_operation_status("Starting dataset split...", "info")
            self.log("🚀 Split dataset button clicked", 'info')
            
            # First save current config
            save_result = self.save_config()
            if not save_result.get('success'):
                error_msg = f"Cannot split: {save_result.get('message', 'Config save failed')}"
                self.update_operation_status(error_msg, "error")
                self.log(f"❌ {error_msg}", 'error')
                return
            
            # TODO: Implement actual dataset splitting logic
            self.log("📊 Dataset split functionality not yet implemented", 'warning')
            self.update_operation_status("Dataset split functionality coming soon", "warning")
            
        except Exception as e:
            error_msg = f"Split dataset error: {e}"
            self.update_operation_status(error_msg, "error")
            self.log(f"❌ {error_msg}", 'error')
    
    # Log method is now provided by LoggingMixin - removed duplicate implementation
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the split module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Initialize using base class which handles everything
            success = super().initialize()
            
            if success and self._config_handler:
                # Set UI components in config handler for extraction/updates
                self._config_handler.set_ui_components(self._ui_components)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize split module: {e}")
            return False
    
    # get_config method is now provided by ConfigurationMixin - removed duplicate
    
    # get_ui_components method is now provided by DisplayMixin - removed duplicate
    
    # get_main_widget method is now provided by DisplayMixin - removed duplicate
    
    
    
    # ==================== CONFIGURATION OPERATIONS ====================
    
    # save_config method is now provided by ConfigurationMixin - removed duplicate
    
    # reset_config method is now provided by ConfigurationMixin - removed duplicate
    
    def get_split_status(self) -> Dict[str, Any]:
        """
        Get current split configuration status.
        
        Returns:
            Status information dictionary
        """
        try:
            if not self._is_initialized:
                return {'initialized': False, 'message': 'Module not initialized'}
            
            config = self.get_config()
            split_config = config.get('split', {})
            ratios = split_config.get('ratios', {})
            
            # Calculate status information
            ratios_sum = sum(ratios.values()) if ratios else 0
            ratios_valid = 0.999 <= ratios_sum <= 1.001
            
            return {
                'initialized': True,
                'module_name': self.module_name,
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components),
                'ratios': ratios,
                'ratios_sum': ratios_sum,
                'ratios_valid': ratios_valid,
                'input_dir': split_config.get('input_dir', ''),
                'output_dir': split_config.get('output_dir', ''),
                'split_method': split_config.get('method', 'random'),
                'seed': split_config.get('seed', 42)
            }
            
        except Exception as e:
            return {'error': f'Status check failed: {str(e)}'}
    
    # _setup_ui_logging_bridge method is now provided by LoggingMixin - removed duplicate
    
    def _initialize_status_panel(self) -> None:
        """Initialize status panel display."""
        try:
            self.update_operation_status("Ready for dataset split configuration", "info")
            self.logger.debug("📊 Status panel initialized")
            
        except Exception as e:
            self.logger.debug(f"Status panel initialization failed: {e}")
    
    def _initialize_progress_display(self) -> None:
        """Initialize operation container and show initial logs like backbone."""
        try:
            # Show rich initialization logs like backbone module
            self.log("🔧 Split module operation manager initialized", 'info')
            self.log("📊 Split UI module ready", 'info')
            
            # Log current configuration status
            config = self.get_config()
            split_config = config.get('split', {})
            ratios = split_config.get('ratios', {})
            
            self.log(f"🎯 Default split ratios loaded: Train={ratios.get('train', 0.7):.1%}, Val={ratios.get('val', 0.15):.1%}, Test={ratios.get('test', 0.15):.1%}", 'info')
            self.log(f"📁 Input directory: {split_config.get('input_dir', 'data/raw')}", 'info')
            self.log(f"📂 Output directory: {split_config.get('output_dir', 'data/split')}", 'info')
            self.log(f"🎲 Random seed: {split_config.get('seed', 42)}", 'debug')
            
            # Show operation container is ready (no progress tracker as requested)
            operation_container = self._ui_components.get('operation_container')
            if operation_container:
                self.log("📈 Operation container ready for split configuration", 'info')
            
            self.log("✅ Split module fully initialized and ready", 'info')
            
        except Exception as e:
            self.logger.debug(f"Operation container initialization failed: {e}")
    
    # _update_status method is now provided by OperationMixin as update_operation_status - removed duplicate
    
    # _cleanup_ui_logging_bridge method is now handled by LoggingMixin - removed duplicate

    # cleanup method is now provided by base classes - removed duplicate


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_split_uimodule_instance: Optional[SplitUIModule] = None


def create_split_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> SplitUIModule:
    """
    Create a new split UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        SplitUIModule instance
    """
    module = SplitUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def get_split_uimodule() -> Optional[SplitUIModule]:
    """
    Get existing split UIModule instance.
    
    Returns:
        Existing SplitUIModule instance or None
    """
    return _split_uimodule_instance


def reset_split_uimodule() -> None:
    """Reset the split UIModule singleton instance."""
    global _split_uimodule_instance
    
    if _split_uimodule_instance:
        _split_uimodule_instance.cleanup()
        _split_uimodule_instance = None


# ==================== SHARED METHODS REGISTRATION ====================

def register_split_shared_methods() -> None:
    """Register shared methods for split module."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register split-specific shared methods
        SharedMethodRegistry.register_method(
            'split.save_config',
            lambda: create_split_uimodule().save_config(),
            description='Save split configuration'
        )
        
        SharedMethodRegistry.register_method(
            'split.reset_config', 
            lambda: create_split_uimodule().reset_config(),
            description='Reset split configuration'
        )
        
        SharedMethodRegistry.register_method(
            'split.get_split_status',
            lambda: create_split_uimodule().get_split_status(),
            description='Get split configuration status'
        )
        
        logger = get_module_logger("smartcash.ui.dataset.split.shared")
        logger.debug("📋 Registered split shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.split.shared")
        logger.error(f"Failed to register shared methods: {e}")


# ==================== CONVENIENCE FUNCTIONS ====================

# Create standardized display function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import create_display_function

initialize_split_ui = create_display_function(
    module_class=SplitUIModule,
    function_name="initialize_split_ui"
)


# Create standardized component function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import create_component_function

get_split_components = create_component_function(
    module_class=SplitUIModule,
    function_name="get_split_components"
)


# ==================== MODULE REGISTRATION ====================

# Auto-register when module is imported
try:
    register_split_shared_methods()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")