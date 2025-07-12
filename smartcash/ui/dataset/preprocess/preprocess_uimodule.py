"""
File: smartcash/ui/dataset/preprocess/preprocess_uimodule.py
Description: Main UIModule implementation for preprocessing module with backend integration
"""

from typing import Dict, Any, Optional, Union
import asyncio
from smartcash.ui.core.ui_module import UIModule, SharedMethodRegistry, register_operation_method
from smartcash.ui.core.ui_module_factory import UIModuleFactory, create_template
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.logger import get_module_logger

# Import preprocessing components
from smartcash.ui.dataset.preprocess.components.preprocess_ui import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocess.configs.preprocess_config_handler import PreprocessConfigHandler
from smartcash.ui.dataset.preprocess.configs.preprocess_defaults import get_default_preprocessing_config
from smartcash.ui.dataset.preprocess.handlers.preprocess_ui_handler import PreprocessUIHandler
from smartcash.ui.dataset.preprocess.constants import UI_CONFIG, MODULE_METADATA

# Import backend service
from smartcash.ui.dataset.preprocess.services.preprocess_service import PreprocessService

# Import operation manager
from smartcash.ui.dataset.preprocess.operations.manager import PreprocessOperationManager


class PreprocessUIModule(UIModule):
    """
    Main UIModule implementation for preprocessing module.
    
    Features:
    - 🎯 UIModule pattern consistency with core modules
    - 📊 YOLO preprocessing with normalization and validation
    - 🔧 Backend service integration for processing operations
    - 🔄 Operation manager for preprocess, check, cleanup operations
    - 📱 Enhanced error handling and logging
    - ♻️ Proper resource management and cleanup
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing UIModule.
        
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
        self._ui_handler = None
        self._operation_manager = None
        self._backend_service = None
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
            default_config = get_default_preprocessing_config()
            
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
        Initialize the preprocessing module.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            self.logger.info("Preprocessing module already initialized")
            return True
        
        try:
            self.logger.info("🎯 Initializing preprocessing module")
            
            # 1. Create configuration handler
            self._config_handler = PreprocessConfigHandler()
            self._config_handler.update_config(self.merged_config)
            
            # 2. Create backend service
            self._backend_service = PreprocessService()
            
            # 3. Create UI components
            self._ui_components = create_preprocessing_main_ui(config=self.merged_config)
            if not self._ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # 4. Create operation manager
            operation_container = self._ui_components.get('operation_container')
            if operation_container:
                self._operation_manager = PreprocessOperationManager(
                    config=self.merged_config,
                    operation_container=operation_container
                )
                # Store in components for handler access
                self._ui_components['operation_manager'] = self._operation_manager
            
            # 5. Create UI handler
            self._ui_handler = PreprocessUIHandler(
                ui_components=self._ui_components,
                config_handler=self._config_handler,
                module_name=self.module_name,
                parent_module=self.parent_module
            )
            
            # 6. Setup event handlers
            self._setup_event_handlers()
            
            # 7. Mark as initialized
            self._is_initialized = True
            self.logger.info("✅ Preprocessing module initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize preprocessing module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._initialization_error = error_msg
            return False
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            if not self._ui_handler:
                return
            
            # Connect button handlers
            preprocess_btn = self._ui_components.get('preprocess_btn')
            if preprocess_btn:
                preprocess_btn.on_click(lambda _: self._ui_handler.handle_preprocess_click())
            
            check_btn = self._ui_components.get('check_btn')
            if check_btn:
                check_btn.on_click(lambda _: self._ui_handler.handle_check_click())
            
            cleanup_btn = self._ui_components.get('cleanup_btn')
            if cleanup_btn:
                cleanup_btn.on_click(lambda _: self._ui_handler.handle_cleanup_click())
            
            # Setup configuration change handlers
            if hasattr(self._ui_handler, 'setup_config_handlers'):
                self._ui_handler.setup_config_handlers(self._ui_components)
            
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
        return components.get('main_container') or components.get('ui')
    
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
    
    # ==================== PREPROCESSING OPERATIONS ====================
    
    def execute_preprocess(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute preprocessing operation.
        
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
            
            # Execute via operation manager if available
            if self._operation_manager:
                return self._operation_manager.execute_preprocess(operation_config)
            
            # Fallback to UI handler
            if self._ui_handler:
                return self._ui_handler.preprocess_dataset(operation_config)
            
            # Fallback to backend service
            if self._backend_service:
                return self._backend_service.preprocess_dataset(operation_config)
            
            return {'success': False, 'message': 'No processing handler available'}
            
        except Exception as e:
            error_msg = f"Preprocessing execution failed: {str(e)}"
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
            
            # Execute via operation manager if available
            if self._operation_manager:
                return self._operation_manager.execute_check(operation_config)
            
            # Fallback to UI handler
            if self._ui_handler:
                return self._ui_handler.get_preprocessing_status(operation_config)
            
            # Fallback to backend service
            if self._backend_service:
                return self._backend_service.get_preprocessing_status(operation_config)
            
            return {'success': False, 'message': 'No check handler available'}
            
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
            
            # Execute via operation manager if available
            if self._operation_manager:
                return self._operation_manager.execute_cleanup(operation_config)
            
            # Fallback to UI handler
            if self._ui_handler:
                return self._ui_handler.cleanup_preprocessing_files(operation_config)
            
            # Fallback to backend service
            if self._backend_service:
                return self._backend_service.cleanup_preprocessing_files(operation_config)
            
            return {'success': False, 'message': 'No cleanup handler available'}
            
        except Exception as e:
            error_msg = f"Cleanup execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """
        Get current preprocessing status.
        
        Returns:
            Status information dictionary
        """
        try:
            if not self._is_initialized:
                return {'initialized': False, 'message': 'Module not initialized'}
            
            # Get status from backend service
            if self._backend_service:
                service_status = self._backend_service.get_preprocessing_status(self.get_config())
            else:
                service_status = {'service_available': False}
            
            return {
                'initialized': True,
                'module_name': self.module_name,
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components),
                'operation_manager_ready': self._operation_manager is not None,
                'backend_service_ready': self._backend_service is not None,
                **service_status
            }
            
        except Exception as e:
            return {'error': f'Status check failed: {str(e)}'}
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            self.logger.info("Cleaning up preprocessing module")
            
            # Cleanup handlers
            if self._ui_handler and hasattr(self._ui_handler, 'cleanup'):
                self._ui_handler.cleanup()
            
            if self._operation_manager and hasattr(self._operation_manager, 'cleanup'):
                self._operation_manager.cleanup()
            
            # Reset state
            self._is_initialized = False
            self._initialization_error = None
            self._ui_components.clear()
            
            # Clear references
            self._config_handler = None
            self._ui_handler = None
            self._operation_manager = None
            self._backend_service = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_preprocess_uimodule_instance: Optional[PreprocessUIModule] = None


def create_preprocess_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    reset_existing: bool = False
) -> PreprocessUIModule:
    """
    Factory function to create preprocessing UIModule.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to automatically initialize the module
        reset_existing: Whether to reset existing singleton instance
        
    Returns:
        PreprocessUIModule instance
    """
    global _preprocess_uimodule_instance
    
    # Reset existing instance if requested
    if reset_existing and _preprocess_uimodule_instance:
        _preprocess_uimodule_instance.cleanup()
        _preprocess_uimodule_instance = None
    
    # Create new instance if none exists
    if _preprocess_uimodule_instance is None:
        _preprocess_uimodule_instance = PreprocessUIModule(config=config)
    
    # Initialize if requested and not already initialized
    if auto_initialize and not _preprocess_uimodule_instance._is_initialized:
        _preprocess_uimodule_instance.initialize()
    
    return _preprocess_uimodule_instance


def get_preprocess_uimodule() -> Optional[PreprocessUIModule]:
    """
    Get existing preprocessing UIModule instance.
    
    Returns:
        Existing PreprocessUIModule instance or None
    """
    return _preprocess_uimodule_instance


def reset_preprocess_uimodule() -> None:
    """Reset the preprocessing UIModule singleton instance."""
    global _preprocess_uimodule_instance
    
    if _preprocess_uimodule_instance:
        _preprocess_uimodule_instance.cleanup()
        _preprocess_uimodule_instance = None


# ==================== SHARED METHODS REGISTRATION ====================

def register_preprocess_shared_methods() -> None:
    """Register shared methods for preprocessing module."""
    try:
        # Register preprocessing-specific shared methods
        shared_methods = {
            'execute_preprocess': lambda module, **kwargs: module.execute_preprocess(kwargs.get('config')),
            'execute_check': lambda module, **kwargs: module.execute_check(kwargs.get('config')),
            'execute_cleanup': lambda module, **kwargs: module.execute_cleanup(kwargs.get('config')),
            'get_preprocessing_status': lambda module: module.get_preprocessing_status(),
            'update_preprocessing_config': lambda module, **kwargs: module.update_config(kwargs.get('config', {}))
        }
        
        # Register each method individually
        for method_name, method_func in shared_methods.items():
            register_operation_method(f"preprocess.{method_name}", method_func)
        
        logger = get_module_logger("smartcash.ui.dataset.preprocess.shared")
        logger.debug("📋 Registered preprocessing shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.preprocess.shared")
        logger.error(f"Failed to register shared methods: {e}")


def register_preprocess_template() -> None:
    """Register preprocessing module template with UIModuleFactory."""
    try:
        template = create_template(
            module_name="preprocess",
            parent_module="dataset",
            default_config=get_default_preprocessing_config(),
            required_components=[
                "main_container", "header_container", "form_container", 
                "action_container", "operation_container", "footer_container"
            ],
            required_operations=[
                "preprocess", "check", "cleanup", "get_preprocessing_status"
            ],
            auto_initialize=False,
            description="Dataset preprocessing module with YOLO normalization"
        )
        
        UIModuleFactory.register_template(template, overwrite=True)
        logger = get_module_logger("smartcash.ui.dataset.preprocess.template")
        logger.debug("📋 Registered preprocessing template")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.dataset.preprocess.template")
        logger.error(f"Failed to register template: {e}")


# ==================== BACKWARD COMPATIBILITY ====================

def initialize_preprocess_ui_uimodule(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True,
    **kwargs
) -> Union[PreprocessUIModule, None]:
    """
    Initialize and optionally display preprocessing UI using UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI (requires IPython)
        **kwargs: Additional arguments
        
    Returns:
        PreprocessUIModule instance if successful, None otherwise
    """
    try:
        # Create and initialize module
        module = create_preprocess_uimodule(config=config, auto_initialize=True)
        
        if display:
            try:
                from IPython.display import display as ipython_display
                main_widget = module.get_main_widget()
                if main_widget:
                    ipython_display(main_widget)
                else:
                    print("⚠️ No UI widget available for display")
            except ImportError:
                print("⚠️ IPython not available, cannot display UI")
            except Exception as e:
                print(f"⚠️ Display failed: {e}")
        
        return module
        
    except Exception as e:
        print(f"❌ Failed to initialize preprocessing UI: {e}")
        return None


def get_preprocess_components_uimodule() -> Dict[str, Any]:
    """
    Get preprocessing UI components using UIModule pattern.
    
    Returns:
        Dictionary of UI components
    """
    module = get_preprocess_uimodule()
    if module:
        return module.get_ui_components()
    return {}


# ==================== MODULE REGISTRATION ====================

# Auto-register when module is imported
try:
    register_preprocess_shared_methods()
    register_preprocess_template()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")