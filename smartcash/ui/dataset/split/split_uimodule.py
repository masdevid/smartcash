"""
File: smartcash/ui/dataset/split/split_uimodule.py
Description: Main UIModule implementation for split configuration module
"""

from typing import Dict, Any, Optional

# Import ipywidgets for UI components
import ipywidgets as widgets

from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Import split components
from smartcash.ui.dataset.split.components.split_ui import create_split_ui
from smartcash.ui.dataset.split.configs.split_config_handler import SplitConfigHandler
from smartcash.ui.dataset.split.configs.split_defaults import get_default_split_config
from smartcash.ui.dataset.split.constants import UI_CONFIG, MODULE_METADATA


class SplitUIModule(UIModule):
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
        
        self.logger = get_module_logger("smartcash.ui.dataset.split")
        
        # Initialize components
        self._config_handler = None
        self._ui_components = None
        self._merged_config = {}
        
        self.logger.debug("✅ SplitUIModule initialized")
    
    def _merge_with_defaults(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with default values.
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Merged configuration dictionary
        """
        try:
            default_config = get_default_split_config()
            
            # Deep merge configurations
            merged = default_config.copy()
            
            def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
                for key, value in override.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        base[key] = deep_merge(base[key], value)
                    else:
                        base[key] = value
                return base
            
            return deep_merge(merged, user_config)
            
        except Exception as e:
            self.logger.error(f"Error merging configurations: {e}")
            return user_config
    
    def _initialize_config_handler(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration handler."""
        try:
            # Get default config first
            default_config = get_default_split_config()
            
            # Merge provided config with defaults
            if config:
                merged_config = self._merge_with_defaults(config)
            else:
                merged_config = default_config
            
            # Initialize config handler
            self._config_handler = SplitConfigHandler(merged_config)
            
            # Store merged config internally
            self._merged_config = merged_config
            
            self.logger.debug(f"✅ Config handler initialized with split configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize config handler: {e}", exc_info=True)
            raise
    
    def _create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components."""
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
    
    def _setup_button_handlers(self) -> None:
        """Setup button event handlers for split operations."""
        try:
            if not self._ui_components:
                self.logger.warning("Cannot setup button handlers - missing UI components")
                return
                
            # Get buttons
            save_button = self._ui_components.get('save_button')
            reset_button = self._ui_components.get('reset_button')
            
            # Setup save button handler
            if save_button and hasattr(save_button, 'on_click'):
                save_button.on_click(self._handle_save_config)
                self.logger.debug("✅ Bound Save button to _handle_save_config")
            else:
                self.logger.warning("Save button not found or doesn't support on_click")
                
            # Setup reset button handler  
            if reset_button and hasattr(reset_button, 'on_click'):
                reset_button.on_click(self._handle_reset_config)
                self.logger.debug("✅ Bound Reset button to _handle_reset_config")
            else:
                self.logger.warning("Reset button not found or doesn't support on_click")
                
        except Exception as e:
            self.logger.error(f"Failed to setup button handlers: {e}")
    
    def _handle_save_config(self, button=None):
        """Handle save config button click."""
        try:
            self.log("💾 Save config button clicked", 'info')
            result = self.save_config()
            if result.get('success'):
                self.log(f"✅ Configuration saved: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Save failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Save config error: {e}", 'error')
    
    def _handle_reset_config(self, button=None):
        """Handle reset config button click."""
        try:
            self.log("🔄 Reset config button clicked", 'info')
            result = self.reset_config()
            if result.get('success'):
                self.log(f"✅ Configuration reset: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Reset failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Reset config error: {e}", 'error')
    
    def log(self, message: str, level: str = 'info') -> None:
        """Log message to operation container."""
        try:
            # Look for operation container with log capability
            operation_container = self._ui_components.get('operation_container')
            if operation_container and isinstance(operation_container, dict):
                log_message_func = operation_container.get('log_message')
                if log_message_func and callable(log_message_func):
                    log_message_func(message, level)
                    return
            
            # Fallback to logger
            getattr(self.logger, level, self.logger.info)(message)
            
        except Exception as e:
            self.logger.error(f"Failed to log message: {e}")
            getattr(self.logger, level, self.logger.info)(message)
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize the split module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
        """
        try:
            # Initialize configuration handler
            self._initialize_config_handler(config)
            
            # Create UI components
            self._ui_components = self._create_ui_components(self._merged_config)
            
            # Setup button event handlers
            self._setup_button_handlers()
            
            # Log successful initialization to operation container
            self.log("✅ Split module initialized successfully", 'info')
            self.log("📊 Ready for dataset split configuration", 'info')
            
            # Call base class initialization to set status to READY
            super().initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize split module: {e}")
            raise RuntimeError("Failed to initialize split module")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        if self._config_handler:
            return self._config_handler.get_config()
        return self._merged_config.copy()
    
    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            Dictionary of UI components
        """
        return self._ui_components or {}
    
    def get_main_widget(self):
        """
        Get main widget for display.
        
        Returns:
            Main UI widget that can be displayed in a notebook
        """
        try:
            components = self.get_ui_components()
            
            # Try to get the main container directly
            main_widget = components.get('main_container')
            
            # If not found, try common container names
            if main_widget is None and 'containers' in components:
                main_widget = components['containers'].get('main')
                
            # If still not found, look for any container-like widget
            if main_widget is None:
                for key, widget in components.items():
                    if (isinstance(widget, (widgets.Widget, widgets.DOMWidget)) and 
                        hasattr(widget, 'layout') and 
                        hasattr(widget, 'children')):
                        main_widget = widget
                        break
            
            # If we have a valid widget, ensure it's properly initialized
            if main_widget is not None:
                # For MainContainer, we need to get its container attribute
                if hasattr(main_widget, 'container'):
                    main_widget = main_widget.container
                
                # Ensure the widget has a layout
                if not hasattr(main_widget, 'layout'):
                    main_widget.layout = widgets.Layout()
                
                # Set a reasonable default size
                if not hasattr(main_widget.layout, 'width'):
                    main_widget.layout.width = '100%'
                if not hasattr(main_widget.layout, 'height'):
                    main_widget.layout.height = 'auto'
                
                return main_widget
            
            # If no main widget found, create a VBox with all components
            from ipywidgets import VBox
            children = [
                v for v in components.values() 
                if v is not None and isinstance(v, (widgets.Widget, widgets.DOMWidget))
            ]
            
            if children:
                container = VBox(children=children, layout=widgets.Layout(width='100%'))
                return container
            
            self.logger.warning("Could not find or create a displayable main widget")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting main widget: {str(e)}")
            
            # Try to return something meaningful even if there was an error
            try:
                from ipywidgets import HTML
                return HTML(f"<div style='color: red; padding: 10px;'>Error displaying UI: {str(e)}</div>")
            except:
                return None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        if self._config_handler:
            return self._config_handler.get_config()
        return self._merged_config.copy()
    
    
    # ==================== CONFIGURATION OPERATIONS ====================
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save current configuration from UI.
        
        Returns:
            Save operation result
        """
        try:
            if not self.is_ready():
                self.initialize()
            
            if not self._config_handler:
                return {'success': False, 'message': 'Configuration handler not available'}
            
            # Extract current config from UI
            current_config = self._config_handler.extract_config_from_ui(self._ui_components)
            
            # Validate configuration
            if not self._config_handler.validate_config(current_config):
                return {'success': False, 'message': 'Configuration validation failed'}
            
            # Update internal config
            self._merged_config = current_config
            self._config_handler.update_config(current_config)
            
            self.logger.info("💾 Configuration saved successfully")
            
            return {
                'success': True,
                'message': 'Configuration saved successfully',
                'config': current_config
            }
            
        except ValueError as e:
            # Handle validation errors specifically
            error_msg = f"Configuration validation error: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
            
        except Exception as e:
            error_msg = f"Save configuration failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Reset operation result
        """
        try:
            if not self.is_ready():
                self.initialize()
            
            # Get default configuration
            default_config = get_default_split_config()
            
            # Update config handler
            if self._config_handler:
                self._config_handler.update_config(default_config)
                
                # Update UI from default config
                if self._ui_components:
                    self._config_handler.update_ui_from_config(self._ui_components, default_config)
            
            # Update internal config
            self._merged_config = default_config
            
            self.logger.info("🔄 Configuration reset to defaults")
            
            return {
                'success': True,
                'message': 'Configuration reset to defaults',
                'config': default_config
            }
            
        except Exception as e:
            error_msg = f"Reset configuration failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
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
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            self.logger.info("Cleaning up split module")
            
            # Reset state
            self._is_initialized = False
            self._initialization_error = None
            self._ui_components.clear()
            
            # Clear references
            self._config_handler = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


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

def initialize_split_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Initialize and optionally display split UI using new UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI (requires IPython)
        
    Returns:
        If display=True: Returns None (displays UI directly)
        If display=False: Returns a dictionary with UI components and status
    """
    try:
        # Create module instance
        module = create_split_uimodule(config=config, auto_initialize=True)
        ui_components = module.get_ui_components()
        
        # Prepare the result dictionary
        result = {
            'success': True,
            'module': module,
            'ui_components': ui_components,
            'status': module.get_split_status()
        }
        
        # Display the UI if requested
        if display and ui_components:
            from IPython.display import display as ipython_display
            
            # Get the main UI container and display it
            main_ui = ui_components.get('main_container')
            if main_ui is not None:
                ipython_display(main_ui)
                return None  # Don't return data when display=True
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to initialize split UI: {str(e)}"
        import logging
        logging.getLogger(__name__).error(error_msg, exc_info=True)
        
        return {
            'success': False,
            'error': error_msg,
            'module': None,
            'ui_components': {},
            'status': {}
        }


def get_split_components() -> Dict[str, Any]:
    """
    Get split UI components.
    
    Returns:
        Dictionary of UI components
    """
    module = get_split_uimodule()
    if module:
        return module.get_ui_components()
    return {}


# ==================== MODULE REGISTRATION ====================

# Auto-register when module is imported
try:
    register_split_shared_methods()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")