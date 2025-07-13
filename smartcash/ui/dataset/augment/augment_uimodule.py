"""
File: smartcash/ui/dataset/augment/augment_uimodule.py
Description: Main UIModule implementation for augmentation module with preserved UI and backend flow
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule, register_operation_method
from smartcash.ui.core.ui_module_factory import UIModuleFactory, create_template
from smartcash.ui.logger import get_module_logger

# Import augmentation components
from smartcash.ui.dataset.augment.components.augment_ui import create_augment_ui
from smartcash.ui.dataset.augment.configs.augment_config_handler import AugmentConfigHandler
from smartcash.ui.dataset.augment.configs.augment_defaults import get_default_augment_config
from smartcash.ui.dataset.augment.constants import UI_CONFIG, MODULE_METADATA

# Import operation manager (will be created)
from smartcash.ui.dataset.augment.operations.augment_operation_manager import AugmentOperationManager


class AugmentUIModule(UIModule):
    """
    Main UIModule implementation for augmentation module.
    
    Features:
    - 🎯 UIModule pattern consistency with core modules
    - 🎨 Complete data augmentation with position and lighting transforms
    - 🔧 Backend service integration for augmentation operations
    - 🔄 Operation manager for augment, check, cleanup, preview operations
    - 📱 Enhanced error handling and logging
    - ♻️ Proper resource management and cleanup
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentation UIModule.
        
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
            default_config = get_default_augment_config()
            
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
        Initialize the augmentation module.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            self.logger.info("Augmentation module already initialized")
            return True
        
        try:
            self.logger.info("🎨 Initializing augmentation module")
            
            # 1. Create configuration handler
            self._config_handler = AugmentConfigHandler()
            self._config_handler.update_config(self.merged_config)
            
            # 2. Create UI components
            self._ui_components = create_augment_ui(config=self.merged_config)
            if not self._ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # 3. Create operation manager
            operation_container = self._ui_components.get('operation_container')
            if operation_container:
                self._operation_manager = AugmentOperationManager(
                    config=self.merged_config,
                    operation_container=operation_container
                )
                # Store in components for handler access
                self._ui_components['operation_manager'] = self._operation_manager
            
            # 4. Setup event handlers
            self._setup_event_handlers()
            
            # 5. Mark as initialized
            self._is_initialized = True
            self.logger.info("✅ Augmentation module initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize augmentation module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._initialization_error = error_msg
            return False
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            if not self._operation_manager:
                return
            
            # Connect button handlers
            augment_btn = self._ui_components.get('augment_button')
            if augment_btn:
                augment_btn.on_click(lambda _: self.execute_augment())
            
            check_btn = self._ui_components.get('check_button')
            if check_btn:
                check_btn.on_click(lambda _: self.execute_check())
            
            cleanup_btn = self._ui_components.get('cleanup_button')
            if cleanup_btn:
                cleanup_btn.on_click(lambda _: self.execute_cleanup())
            
            preview_btn = self._ui_components.get('preview_button')
            if preview_btn:
                preview_btn.on_click(lambda _: self.execute_preview())
            
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
            Main UI widget
        """
        components = self.get_ui_components()
        # Try to get the main container from various possible locations
        return (components.get('ui_components', {}).get('main_container') or
                components.get('main_container') or
                components.get('ui') or
                components.get('container'))
    
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
            
            self.logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    # ==================== AUGMENTATION OPERATIONS ====================
    
    def execute_augment(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute augmentation operation.
        
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
                return self._operation_manager.execute_augment(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Augmentation execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_check(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute dataset check operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Check result dictionary
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Use provided config or current config
            operation_config = config or self.get_config()
            
            # Execute via operation manager
            if self._operation_manager:
                return self._operation_manager.execute_check(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Check execution failed: {str(e)}"
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
    
    def execute_preview(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute preview operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Preview result dictionary
        """
        try:
            if not self._is_initialized and not self.initialize():
                return {'success': False, 'message': 'Module not initialized'}
            
            # Use provided config or current config
            operation_config = config or self.get_config()
            
            # Execute via operation manager
            if self._operation_manager:
                return self._operation_manager.execute_preview(operation_config)
            
            return {'success': False, 'message': 'No operation manager available'}
            
        except Exception as e:
            error_msg = f"Preview execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """
        Get current augmentation status.
        
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
            
            return {
                'initialized': True,
                'module_name': self.module_name,
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components),
                'operation_manager_ready': self._operation_manager is not None,
                **manager_status
            }
            
        except Exception as e:
            return {'error': f'Status check failed: {str(e)}'}
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            self.logger.info("Cleaning up augmentation module")
            
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
_augment_uimodule_instance: Optional[AugmentUIModule] = None


def create_augment_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    reset_existing: bool = False
) -> AugmentUIModule:
    """
    Factory function to create augmentation UIModule.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to automatically initialize the module
        reset_existing: Whether to reset existing singleton instance
        
    Returns:
        AugmentUIModule instance
    """
    global _augment_uimodule_instance
    
    # Reset existing instance if requested
    if reset_existing and _augment_uimodule_instance:
        _augment_uimodule_instance.cleanup()
        _augment_uimodule_instance = None
    
    # Create new instance if none exists
    if _augment_uimodule_instance is None:
        _augment_uimodule_instance = AugmentUIModule(config=config)
    
    # Initialize if requested and not already initialized
    if auto_initialize and not _augment_uimodule_instance._is_initialized:
        _augment_uimodule_instance.initialize()
    
    return _augment_uimodule_instance


def get_augment_uimodule() -> Optional[AugmentUIModule]:
    """
    Get existing augmentation UIModule instance.
    
    Returns:
        Existing AugmentUIModule instance or None
    """
    return _augment_uimodule_instance


def reset_augment_uimodule() -> None:
    """Reset the augmentation UIModule singleton instance."""
    global _augment_uimodule_instance
    
    if _augment_uimodule_instance:
        _augment_uimodule_instance.cleanup()
        _augment_uimodule_instance = None


# ==================== SHARED METHODS REGISTRATION ====================

def register_augment_shared_methods() -> None:
    """Register shared methods for augmentation module."""
    try:
        # Register augmentation-specific shared methods
        shared_methods = {
            'execute_augment': lambda module, **kwargs: module.execute_augment(kwargs.get('config')),
            'execute_check': lambda module, **kwargs: module.execute_check(kwargs.get('config')),
            'execute_cleanup': lambda module, **kwargs: module.execute_cleanup(kwargs.get('config')),
            'execute_preview': lambda module, **kwargs: module.execute_preview(kwargs.get('config')),
            'get_augmentation_status': lambda module: module.get_augmentation_status(),
            'update_augmentation_config': lambda module, **kwargs: module.update_config(kwargs.get('config', {}))
        }
        
        # Register each method individually
        for method_name, method_func in shared_methods.items():
            register_operation_method(f"augment.{method_name}", method_func)
        
        logger = get_module_logger("smartcash.ui.dataset.augment.shared")
        logger.debug("📋 Registered augmentation shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.augment.shared")
        logger.error(f"Failed to register shared methods: {e}")


def register_augment_template() -> None:
    """Register augmentation module template with UIModuleFactory."""
    try:
        template = create_template(
            module_name="augment",
            parent_module="dataset",
            default_config=get_default_augment_config(),
            required_components=[
                "main_container", "header_container", "form_container", 
                "action_container", "operation_container", "footer_container"
            ],
            required_operations=[
                "augment", "check", "cleanup", "preview", "get_augmentation_status"
            ],
            auto_initialize=False,
            description="Dataset augmentation module with position and lighting transforms"
        )
        
        UIModuleFactory.register_template(template, overwrite=True)
        logger = get_module_logger("smartcash.ui.dataset.augment.template")
        logger.debug("📋 Registered augmentation template")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.augment.template")
        logger.error(f"Failed to register template: {e}")


# ==================== CONVENIENCE FUNCTIONS ====================

def initialize_augment_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True
) -> Dict[str, Any]:
    """
    Initialize and optionally display augmentation UI using UIModule pattern.
    
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
        module = create_augment_uimodule(config=config, auto_initialize=True)
        
        if not module or not hasattr(module, '_is_initialized') or not module._is_initialized:
            error_msg = "Failed to initialize augment module"
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
                from IPython.display import display as ipython_display
                
                # Try to get the main widget first
                main_widget = module.get_main_widget() if hasattr(module, 'get_main_widget') else None
                
                if main_widget is not None:
                    ipython_display(main_widget)
                else:
                    # Fall back to UI components if main widget is not available
                    if 'ui_components' in ui_components and 'ui' in ui_components['ui_components']:
                        ipython_display(ui_components['ui_components']['ui'])
                    elif 'ui' in ui_components:
                        ipython_display(ui_components['ui'])
                    
            except ImportError:
                print("⚠️ IPython not available, cannot display UI")
            except Exception as e:
                print(f"⚠️ Display failed: {str(e)}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to initialize augmentation UI: {str(e)}"
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


def get_augment_components() -> Dict[str, Any]:
    """
    Get augmentation UI components.
    
    Returns:
        Dictionary of UI components
    """
    module = get_augment_uimodule()
    if module:
        return module.get_ui_components()
    return {}


# ==================== MODULE REGISTRATION ====================

# Auto-register when module is imported
try:
    register_augment_shared_methods()
    register_augment_template()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")