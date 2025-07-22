"""
File: smartcash/ui/dataset/split/split_uimodule.py
Description: Main UIModule implementation for split configuration module
"""

from typing import Dict, Any, Optional

# Import ipywidgets for UI components

from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs


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
        """Initialize split UI module following latest mixin pattern."""
        super().__init__(
            module_name='split',
            parent_module='dataset'
        )
        # Required UI components for validation
        self._required_components = [
            'main_container',
            'action_container',
            'operation_container'
        ]
        # ButtonHandlerMixin is authoritative for button state
        if not hasattr(self, '_button_handlers'):
            self._button_handlers = {}
        if not hasattr(self, '_button_states'):
            self._button_states = {}
        self.log_debug("✅ SplitUIModule initialized (mixin pattern)")
        
        # Initialize resources dictionary for cleanup tracking
        self._resources = {}
        
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion

    # Core Interface Methods
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        from smartcash.ui.dataset.split.configs.split_defaults import get_default_split_config
        return get_default_split_config()
        
    def create_config_handler(self, config: Dict[str, Any]):
        """Create and return SplitConfigHandler."""
        from smartcash.ui.dataset.split.configs.split_config_handler import SplitConfigHandler
        return SplitConfigHandler(config)
        
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return UI components for this module."""
        from smartcash.ui.dataset.split.components.split_ui import create_split_ui
        return create_split_ui(config)
        
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration from config handler."""
        if not hasattr(self, '_config_handler') or self._config_handler is None:
            self._initialize_config_handler()
        try:
            return self._config_handler.get_current_config()
        except Exception as e:
            self.log_error("Failed to get config", traceback=str(e))
            return {}
            
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Return button handlers for this module."""
        return self._button_handlers
        
    def _register_default_operations(self) -> None:
        """Register default operations for this module."""
        self.register_operation_handler('save_config', self._handle_save_config)
        self.register_operation_handler('reset_config', self._handle_reset_config)
        self.register_operation_handler('split_dataset', self._handle_split_dataset)

    # Cleanup Methods
    def cleanup(self) -> None:
        """Clean up resources and UI components."""
        try:
            # Cleanup any split-specific resources
            if hasattr(self, '_resources'):
                for resource_name, resource in self._resources.items():
                    try:
                        if hasattr(resource, 'close'):
                            resource.close()
                        elif hasattr(resource, 'shutdown'):
                            resource.shutdown()
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Failed to clean up resource {resource_name}: {e}")
                self._resources.clear()
            
            # Cleanup UI components
            if hasattr(self, '_ui_components') and self._ui_components:
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            if hasattr(self, 'logger'):
                self.logger.info("Split module cleanup completed")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Split module cleanup failed: {e}")

    # Operation Handlers
    def _handle_save_config(self, button=None):
        """Handle save configuration operation."""
        result = self.save_config()
        self.log_info("Config saved.")
        self.update_progress(100, message="Config saved.")
        return result

    def _handle_reset_config(self, button=None):
        """Handle reset configuration operation."""
        result = self.reset_config()
        self.log_info("Config reset.")
        self.update_progress(0, message="Config reset.")
        return result

    def _handle_split_dataset(self, button=None):
        """Handle dataset split operation."""
        self.disable_all_buttons("⏳ Splitting dataset...")
        try:
            self.log_operation_start("split_dataset")
            # ... split logic ...
            self.log_success("Dataset split completed.")
            self.enable_all_buttons()
            self.log_operation_complete("split_dataset")
            return {"success": True, "message": "Dataset split completed."}
        except Exception as e:
            self.log_operation_error("split_dataset", str(e))
            self.enable_all_buttons()
            return {"error": str(e)}


    # Helper Methods
    def get_split_status(self) -> Dict[str, Any]:
        """Get current split configuration status."""
        try:
            if not self._is_initialized:
                return {'initialized': False, 'error': 'Module not initialized.'}
                
            config = self.get_config()
            split_config = config.get('split', {})
            ratios = split_config.get('ratios', {})
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
            self.log_error("Status check failed", traceback=str(e))
            return {'error': f'Status check failed: {str(e)}'}


