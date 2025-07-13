"""
File: smartcash/ui/dataset/split/split_uimodule.py
Description: Main UIModule implementation for split configuration module
"""

from typing import Dict, Any, Optional

# Import ipywidgets for UI components
import ipywidgets as widgets

from smartcash.ui.core.ui_module import UIModule, register_operation_method
from smartcash.ui.core.ui_module_factory import UIModuleFactory, create_template
from smartcash.ui.logger import get_module_logger

# Import split components
from smartcash.ui.dataset.split.components.split_ui import create_split_ui
from smartcash.ui.dataset.split.configs.split_config_handler import SplitConfigHandler
from smartcash.ui.dataset.split.configs.split_defaults import get_default_split_config
from smartcash.ui.dataset.split.constants import UI_CONFIG, MODULE_METADATA


class SplitUIModule(UIModule):
    """
    Main UIModule implementation for split configuration module.
    
    Features:
    - 📋 Split configuration management (train/val/test ratios)
    - 💾 Save and reset configuration functionality
    - 🎯 UIModule pattern consistency with core modules
    - 🔧 Configuration validation and UI synchronization
    - 📱 Enhanced error handling and logging
    - ♻️ Proper resource management and cleanup
    
    Note: This is a configuration-only module with save/reset buttons.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize split UIModule.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(
            module_name=UI_CONFIG['module_name'],
            parent_module=UI_CONFIG['parent_module']
        )
        
        # Store module metadata
        self.module_metadata = MODULE_METADATA
        
        # Initialize with provided or default config
        self.config = config or {}
        self.merged_config = self._merge_with_defaults(self.config)
        
        # Initialize components
        self._config_handler = None
        self._ui_components = {}
        
        # Track initialization state
        self._is_initialized = False
        self._initialization_error = None
    
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
    
    def initialize(self) -> bool:
        """
        Initialize the split module.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            self.logger.info("Split module already initialized")
            return True
        
        try:
            self.logger.info("📊 Initializing split configuration module")
            
            # 1. Create configuration handler
            self._config_handler = SplitConfigHandler(self.merged_config)
            
            # 2. Create UI components
            self._ui_components = create_split_ui(config=self.merged_config)
            if not self._ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # 3. Setup event handlers for save/reset buttons
            self._setup_event_handlers()
            
            # 4. Mark as initialized
            self._is_initialized = True
            self.logger.info("✅ Split module initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize split module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._initialization_error = error_msg
            return False
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            # Connect save button handler
            save_btn = self._ui_components.get('save_button')
            if save_btn and hasattr(save_btn, 'on_click'):
                save_btn.on_click(lambda _: self.save_config())
            
            # Connect reset button handler
            reset_btn = self._ui_components.get('reset_button')
            if reset_btn and hasattr(reset_btn, 'on_click'):
                reset_btn.on_click(lambda _: self.reset_config())
            
            self.logger.info("✅ Event handlers connected")
            
        except Exception as e:
            self.logger.error(f"Error setting up event handlers: {e}")
    
    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            Dictionary of UI components
        """
        if not self._is_initialized:
            if not self.initialize():
                return {'error': self._initialization_error or 'Failed to initialize'}
        
        return self._ui_components.copy()
    
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
        return self.merged_config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update module configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Merge with existing config
            self.merged_config = self._merge_with_defaults(new_config)
            
            # Update config handler if available
            if self._config_handler:
                self._config_handler.update_config(self.merged_config)
                
                # Update UI from new config
                if self._ui_components:
                    self._config_handler.update_ui_from_config(self._ui_components)
            
            self.logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    # ==================== CONFIGURATION OPERATIONS ====================
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save current configuration from UI.
        
        Returns:
            Save operation result
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            if not self._config_handler:
                return {'success': False, 'message': 'Configuration handler not available'}
            
            # Extract current config from UI
            current_config = self._config_handler.extract_config_from_ui(self._ui_components)
            
            # Validate configuration
            if not self._config_handler.validate_config(current_config):
                return {'success': False, 'message': 'Configuration validation failed'}
            
            # Update internal config
            self.merged_config = current_config
            self._config_handler.update_config(current_config)
            
            self.logger.info("💾 Configuration saved successfully")
            
            # Log to UI if available
            log_accordion = self._ui_components.get('log_accordion')
            if log_accordion and hasattr(log_accordion, 'children') and log_accordion.children:
                log_output = log_accordion.children[0]
                if hasattr(log_output, 'append_stdout'):
                    log_output.append_stdout("✅ Configuration saved successfully\n")
            
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
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Get default configuration
            default_config = get_default_split_config()
            
            # Update config handler
            if self._config_handler:
                self._config_handler.update_config(default_config)
                
                # Update UI from default config
                if self._ui_components:
                    self._config_handler.update_ui_from_config(self._ui_components, default_config)
            
            # Update internal config
            self.merged_config = default_config
            
            self.logger.info("🔄 Configuration reset to defaults")
            
            # Log to UI if available
            log_accordion = self._ui_components.get('log_accordion')
            if log_accordion and hasattr(log_accordion, 'children') and log_accordion.children:
                log_output = log_accordion.children[0]
                if hasattr(log_output, 'append_stdout'):
                    log_output.append_stdout("🔄 Configuration reset to defaults\n")
            
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
    reset_existing: bool = False
) -> SplitUIModule:
    """
    Factory function to create split UIModule.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to automatically initialize the module
        reset_existing: Whether to reset existing singleton instance
        
    Returns:
        SplitUIModule instance
    """
    global _split_uimodule_instance
    
    # Reset existing instance if requested
    if reset_existing and _split_uimodule_instance:
        _split_uimodule_instance.cleanup()
        _split_uimodule_instance = None
    
    # Create new instance if none exists
    if _split_uimodule_instance is None:
        _split_uimodule_instance = SplitUIModule(config=config)
    
    # Initialize if requested and not already initialized
    if auto_initialize and not _split_uimodule_instance._is_initialized:
        _split_uimodule_instance.initialize()
    
    return _split_uimodule_instance


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
        # Register split-specific shared methods
        shared_methods = {
            'save_config': lambda module, **kwargs: module.save_config(),
            'reset_config': lambda module, **kwargs: module.reset_config(),
            'get_split_status': lambda module: module.get_split_status(),
            'update_split_config': lambda module, **kwargs: module.update_config(kwargs.get('config', {}))
        }
        
        # Register each method individually
        for method_name, method_func in shared_methods.items():
            register_operation_method(f"split.{method_name}", method_func)
        
        logger = get_module_logger("smartcash.ui.dataset.split.shared")
        logger.debug("📋 Registered split shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.split.shared")
        logger.error(f"Failed to register shared methods: {e}")


def register_split_template() -> None:
    """Register split module template with UIModuleFactory."""
    try:
        template = create_template(
            module_name="split",
            parent_module="dataset",
            default_config=get_default_split_config(),
            required_components=[
                "main_container", "header_container", "form_container", 
                "action_container", "footer_container"
            ],
            required_operations=[
                "save_config", "reset_config", "get_split_status"
            ],
            auto_initialize=False,
            description="Dataset split configuration module with train/validation/test ratios"
        )
        
        UIModuleFactory.register_template(template, overwrite=True)
        logger = get_module_logger("smartcash.ui.dataset.split.template")
        logger.debug("📋 Registered split template")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.split.template")
        logger.error(f"Failed to register template: {e}")


# ==================== CONVENIENCE FUNCTIONS ====================

def initialize_split_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True
) -> Dict[str, Any]:
    """
    Initialize and optionally display split UI using UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI (requires IPython)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if initialization was successful
        - module: reference to the module instance (None if failed)
        - ui_components: dictionary of UI components (empty if failed)
        - status: current module status (empty if failed)
    """
    try:
        # Create and initialize module
        module = create_split_uimodule(config=config, auto_initialize=True)
        
        if not module or not hasattr(module, '_is_initialized') or not module._is_initialized:
            error_msg = "Failed to initialize split module"
            if hasattr(module, '_initialization_error') and module._initialization_error:
                error_msg += f": {module._initialization_error}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'module': None,
                'ui_components': {},
                'status': {}
            }
        
        # Get UI components and status
        ui_components = module.get_ui_components()
        status = module.get_status() if hasattr(module, 'get_status') else {}
        
        result = {
            'success': True,
            'module': module,
            'ui_components': ui_components,
            'status': status
        }
        
        # Display UI if requested and components are available
        if display:
            try:
                from IPython import get_ipython
                from IPython.display import display as ipython_display
                
                # Get the main widget
                main_widget = module.get_main_widget() if hasattr(module, 'get_main_widget') else None
                
                if main_widget is not None:
                    # Check if we're in a notebook environment
                    in_notebook = False
                    try:
                        in_notebook = get_ipython() is not None
                    except:
                        pass
                    
                    if in_notebook:
                        # In notebook, use IPython display with full width
                        if hasattr(main_widget, 'layout'):
                            try:
                                if not hasattr(main_widget.layout, 'width'):
                                    main_widget.layout.width = '100%'
                            except Exception as e:
                                print(f"⚠️ Could not set widget width: {e}")
                        ipython_display(main_widget)
                    else:
                        # In script, just print the widget info
                        print("Split UI initialized. Running in script mode.")
                        print(f"Widget type: {type(main_widget).__name__}")
                
            except ImportError:
                print("⚠️ IPython not available, cannot display UI")
            except Exception as e:
                print(f"⚠️ Failed to display UI: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to initialize split UI: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
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
    register_split_template()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")