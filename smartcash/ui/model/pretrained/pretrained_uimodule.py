"""
File: smartcash/ui/model/pretrained/pretrained_uimodule.py
Description: Main UIModule implementation for pretrained models module
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule, register_operation_method
from smartcash.ui.core.ui_module_factory import UIModuleFactory, create_template
from smartcash.ui.logger import get_module_logger

# Import pretrained components
from smartcash.ui.model.pretrained.components.pretrained_ui import create_pretrained_ui
from smartcash.ui.model.pretrained.configs.pretrained_config_handler import PretrainedConfigHandler
from smartcash.ui.model.pretrained.configs.pretrained_defaults import get_default_pretrained_config
from smartcash.ui.model.pretrained.constants import UI_CONFIG, MODULE_METADATA

# Import operation manager
from smartcash.ui.model.pretrained.operations.pretrained_operation_manager import PretrainedOperationManager


class PretrainedUIModule(UIModule):
    """
    Main UIModule implementation for pretrained models module.
    
    Features:
    - 🤖 Pretrained model management (YOLOv5s, EfficientNet-B4)
    - 📥 Model download operations with progress tracking
    - 🔍 Model validation and integrity checking
    - 🧹 Cleanup corrupted or invalid models
    - 🔄 Refresh model status and directory contents
    - 🎯 UIModule pattern consistency with core modules
    - 🔧 Backend service integration for model operations
    - 📱 Enhanced error handling and logging
    - ♻️ Proper resource management and cleanup
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pretrained UIModule.
        
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
        self._operation_manager = None
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
            default_config = get_default_pretrained_config()
            
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
        Initialize the pretrained module.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            self.logger.info("Pretrained module already initialized")
            return True
        
        try:
            self.logger.info("🤖 Initializing pretrained models module")
            
            # 1. Create configuration handler
            self._config_handler = PretrainedConfigHandler(self.merged_config)
            
            # 2. Create UI components
            ui_result = create_pretrained_ui(config=self.merged_config)
            if not ui_result:
                raise RuntimeError("Failed to create UI components")
            
            # Store the nested ui_components instead of the flattened result
            self._ui_components = ui_result.get('ui_components', {})
            
            # 3. Create operation manager  
            operation_container = self._ui_components.get('containers', {}).get('operation')
            if operation_container:
                self._operation_manager = PretrainedOperationManager(
                    config=self.merged_config,
                    operation_container=operation_container
                )
                # Store in components for handler access
                self._ui_components['operation_manager'] = self._operation_manager
                self.logger.info("✅ Initialized PretrainedOperationManager")
            
            # 4. Setup event handlers
            self._setup_event_handlers()
            
            # 5. Mark as initialized
            self._is_initialized = True
            self.logger.info("✅ Pretrained models module initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize pretrained module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._initialization_error = error_msg
            return False
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            if not self._operation_manager:
                return
            
            # Connect button handlers based on available buttons (using correct action container structure)
            # Try different access patterns for action container
            action_container = (
                self._ui_components.get('action_container', {}) or 
                self._ui_components.get('containers', {}).get('action', {})
            )
            buttons = action_container.get('buttons', {})
            
            # Download button
            download_btn = buttons.get('download')
            if download_btn and hasattr(download_btn, 'on_click'):
                download_btn.on_click(lambda _: self.execute_download())
            
            # Validate button  
            validate_btn = buttons.get('validate')
            if validate_btn and hasattr(validate_btn, 'on_click'):
                validate_btn.on_click(lambda _: self.execute_validate())
            
            # Cleanup button
            cleanup_btn = buttons.get('cleanup')
            if cleanup_btn and hasattr(cleanup_btn, 'on_click'):
                cleanup_btn.on_click(lambda _: self.execute_cleanup())
            
            # Refresh button
            refresh_btn = buttons.get('refresh')
            if refresh_btn and hasattr(refresh_btn, 'on_click'):
                refresh_btn.on_click(lambda _: self.execute_refresh())
            
            connected_buttons = len([btn for btn in buttons.values() if btn and hasattr(btn, 'on_click')])
            self.logger.info(f"✅ Event handlers connected for {connected_buttons} buttons")
            
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
            Main UI widget
        """
        components = self.get_ui_components()
        
        # Try to get the main container from different possible locations
        if 'ui' in components:
            return components['ui']
        if 'main_container' in components:
            main_container = components['main_container']
            if hasattr(main_container, 'container'):
                return main_container.container
            return main_container
            
        # If we get here, try to find the main container in the UI components
        if 'containers' in components:
            if 'main' in components['containers']:
                return components['containers']['main']
            if 'header' in components['containers'] and hasattr(components['containers']['header'], 'container'):
                return components['containers']['header'].container
                
        # As a last resort, return None
        self.logger.warning("Could not find main widget in UI components")
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
    
    # ==================== PRETRAINED OPERATIONS ====================
    
    def execute_download(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute pretrained models download operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Operation result dictionary
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Use provided config or current config
            operation_config = config or self.get_config()
            
            # Execute via operation manager
            if self._operation_manager:
                return self._operation_manager.execute_download(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Download execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_validate(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute model validation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Validation result dictionary
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Use provided config or current config
            operation_config = config or self.get_config()
            
            # Execute via operation manager
            if self._operation_manager:
                return self._operation_manager.execute_validate(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Validation execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_cleanup(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute cleanup operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Cleanup result dictionary
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Use provided config or current config
            operation_config = config or self.get_config()
            
            # Execute via operation manager
            if self._operation_manager:
                return self._operation_manager.execute_cleanup(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Cleanup execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_refresh(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute refresh operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Refresh result dictionary
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Use provided config or current config
            operation_config = config or self.get_config()
            
            # Execute via operation manager
            if self._operation_manager:
                return self._operation_manager.execute_refresh(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Refresh execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def get_pretrained_status(self) -> Dict[str, Any]:
        """
        Get current pretrained models status.
        
        Returns:
            Status information dictionary
        """
        try:
            if not self._is_initialized:
                return {'initialized': False, 'message': 'Module not initialized'}
            
            # Get status from operation manager
            if self._operation_manager:
                manager_status = self._operation_manager.get_status()
            else:
                manager_status = {'operation_manager_ready': False}
            
            config = self.get_config()
            pretrained_config = config.get('pretrained', {})
            
            return {
                'initialized': True,
                'module_name': self.module_name,
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components),
                'operation_manager_ready': self._operation_manager is not None,
                'models_dir': pretrained_config.get('models_dir', '/data/pretrained'),
                'auto_download': pretrained_config.get('auto_download', False),
                'validate_downloads': pretrained_config.get('validate_downloads', True),
                **manager_status
            }
            
        except Exception as e:
            return {'error': f'Status check failed: {str(e)}'}
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            self.logger.info("Cleaning up pretrained module")
            
            # Cleanup operation manager
            if self._operation_manager and hasattr(self._operation_manager, 'cleanup'):
                self._operation_manager.cleanup()
            
            # Reset state
            self._is_initialized = False
            self._initialization_error = None
            self._ui_components.clear()
            
            # Clear references
            self._config_handler = None
            self._operation_manager = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_pretrained_uimodule_instance: Optional[PretrainedUIModule] = None


def create_pretrained_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    reset_existing: bool = False
) -> PretrainedUIModule:
    """
    Factory function to create pretrained UIModule.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to automatically initialize the module
        reset_existing: Whether to reset existing singleton instance
        
    Returns:
        PretrainedUIModule instance
    """
    global _pretrained_uimodule_instance
    
    # Reset existing instance if requested
    if reset_existing and _pretrained_uimodule_instance:
        _pretrained_uimodule_instance.cleanup()
        _pretrained_uimodule_instance = None
    
    # Create new instance if none exists
    if _pretrained_uimodule_instance is None:
        _pretrained_uimodule_instance = PretrainedUIModule(config=config)
    
    # Initialize if requested and not already initialized
    if auto_initialize and not _pretrained_uimodule_instance._is_initialized:
        _pretrained_uimodule_instance.initialize()
    
    return _pretrained_uimodule_instance


def get_pretrained_uimodule() -> Optional[PretrainedUIModule]:
    """
    Get existing pretrained UIModule instance.
    
    Returns:
        Existing PretrainedUIModule instance or None
    """
    return _pretrained_uimodule_instance


def reset_pretrained_uimodule() -> None:
    """Reset the pretrained UIModule singleton instance."""
    global _pretrained_uimodule_instance
    
    if _pretrained_uimodule_instance:
        _pretrained_uimodule_instance.cleanup()
        _pretrained_uimodule_instance = None


# ==================== SHARED METHODS REGISTRATION ====================

def register_pretrained_shared_methods() -> None:
    """Register shared methods for pretrained module."""
    try:
        # Register pretrained-specific shared methods
        shared_methods = {
            'execute_download': lambda module, **kwargs: module.execute_download(kwargs.get('config')),
            'execute_validate': lambda module, **kwargs: module.execute_validate(kwargs.get('config')),
            'execute_cleanup': lambda module, **kwargs: module.execute_cleanup(kwargs.get('config')),
            'execute_refresh': lambda module, **kwargs: module.execute_refresh(kwargs.get('config')),
            'get_pretrained_status': lambda module: module.get_pretrained_status(),
            'update_pretrained_config': lambda module, **kwargs: module.update_config(kwargs.get('config', {}))
        }
        
        # Register each method individually
        for method_name, method_func in shared_methods.items():
            register_operation_method(f"pretrained.{method_name}", method_func)
        
        logger = get_module_logger("smartcash.ui.model.pretrained.shared")
        logger.debug("📋 Registered pretrained shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.model.pretrained.shared")
        logger.error(f"Failed to register shared methods: {e}")


def register_pretrained_template() -> None:
    """Register pretrained module template with UIModuleFactory."""
    try:
        template = create_template(
            module_name="pretrained",
            parent_module="model",
            default_config=get_default_pretrained_config(),
            required_components=[
                "main_container", "header_container", "form_container", 
                "action_container", "operation_container", "footer_container"
            ],
            required_operations=[
                "download", "validate", "cleanup", "refresh", "get_pretrained_status"
            ],
            auto_initialize=False,
            description="Pretrained models management module with download and validation operations"
        )
        
        UIModuleFactory.register_template(template, overwrite=True)
        logger = get_module_logger("smartcash.ui.model.pretrained.template")
        logger.debug("📋 Registered pretrained template")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.model.pretrained.template")
        logger.error(f"Failed to register template: {e}")



def initialize_pretrained_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Initialize and optionally display pretrained UI using UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI (requires IPython)
        
    Returns:
        If display=True: Returns None (displays UI directly)
        If display=False: Returns a dictionary with UI components and status
    """
    try:
        # Get the module and UI components
        module = create_pretrained_uimodule(config=config, auto_initialize=True)
        ui_components = module.get_ui_components()
        
        # Prepare the result dictionary
        result = {
            'success': True,
            'module': module,
            'ui_components': ui_components,
            'status': module.get_pretrained_status()
        }
        
        # Display the UI if requested
        if display and ui_components:
            from IPython import get_ipython
            from IPython.display import display as ipython_display, clear_output
            if get_ipython() is not None:
                clear_output(wait=True)
            # Try different paths to find the main UI container
            main_ui = (
                ui_components.get('main_container') or 
                ui_components.get('ui_components', {}).get('main_container') or
                ui_components.get('ui')
            )
            if main_ui is not None:
                try:
                    if hasattr(main_ui, 'show'):
                        ui_widget = main_ui.show()
                        ipython_display(ui_widget)
                    else:
                        ipython_display(main_ui)
                except Exception as e:
                    module.logger.error(f"Error displaying UI: {str(e)}")
                    ipython_display(main_ui)
                return None  # Don't return data when display=True
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to initialize pretrained UI: {str(e)}"
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


def get_pretrained_components() -> Dict[str, Any]:
    """
    Get pretrained UI components.
    
    Returns:
        Dictionary of UI components
    """
    module = get_pretrained_uimodule()
    if module:
        return module.get_ui_components()
    return {}


# ==================== MODULE REGISTRATION ====================

# Auto-register when module is imported
try:
    register_pretrained_shared_methods()
    register_pretrained_template()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")