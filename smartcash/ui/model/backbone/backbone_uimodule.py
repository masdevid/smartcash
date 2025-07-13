"""
File: smartcash/ui/model/backbone/backbone_uimodule.py
Main UIModule implementation for backbone module following new UIModule pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule, ModuleStatus
from smartcash.ui.logger import get_module_logger
from .configs.backbone_config_handler import BackboneConfigHandler
from .configs.backbone_defaults import get_default_backbone_config
from .operations.backbone_operation_manager import BackboneOperationManager
from datetime import datetime


class BackboneUIModule(UIModule):
    """
    UIModule implementation for backbone configuration.
    
    Features:
    - 🧬 Backbone model selection and configuration
    - 🏗️ Early training pipeline integration
    - 📊 Model summary and statistics generation  
    - 🔧 Configuration validation and management
    - 🎯 Backend model builder integration
    - 📋 Config summary panel in summary_container
    - 🔄 Progress tracking for all operations
    """
    
    def __init__(self):
        """Initialize backbone UI module."""
        super().__init__(
            module_name='backbone',
            parent_module='model'
        )
        
        self.logger = get_module_logger("smartcash.ui.model.backbone")
        
        # Initialize components
        self._config_handler = None
        self._operation_manager = None
        self._ui_components = None
        
        self.logger.debug("✅ BackboneUIModule initialized")
    
    def _initialize_config_handler(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration handler."""
        try:
            self._config_handler = BackboneConfigHandler()
            
            # Set initial configuration
            if config:
                merged_config = self._config_handler.merge_config(
                    get_default_backbone_config(), config
                )
            else:
                merged_config = get_default_backbone_config()
            
            # Update config using keyword arguments
            self.update_config(**merged_config)
            self.logger.debug("✅ Config handler initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize config handler: {e}")
            raise
    
    def _initialize_operation_manager(self) -> None:
        """Initialize operation manager."""
        try:
            if not self._ui_components:
                raise RuntimeError("UI components must be created before operation manager")
            
            operation_container = self._ui_components.get('operation_container')
            if not operation_container:
                raise RuntimeError("Operation container not found in UI components")
            
            self._operation_manager = BackboneOperationManager(
                config=self.get_config(),
                operation_container=operation_container
            )
            
            self._operation_manager.initialize()
            self.logger.debug("✅ Operation manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize operation manager: {e}")
            raise
    
    def _create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components."""
        try:
            from .components.backbone_ui import create_backbone_ui
            
            self.logger.debug("Creating backbone UI components...")
            ui_components = create_backbone_ui(config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize the backbone module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
        """
        try:
            self.logger.info("🧬 Initializing backbone models module")
            
            # Initialize configuration handler
            self._initialize_config_handler(config)
            
            # Create UI components
            self._ui_components = self._create_ui_components(self.get_config())
            
            # Initialize operation manager
            self._initialize_operation_manager()
            
            # Register shared methods for cross-module integration
            self._register_shared_methods()
            
            # Set status to READY to indicate successful initialization
            self._status = ModuleStatus.READY
            self._initialized_at = datetime.now()
            self.logger.info("✅ Backbone models module initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backbone module: {e}")
            raise RuntimeError("Failed to create UI components")
    
    def _register_shared_methods(self) -> None:
        """Register shared methods for cross-module integration."""
        try:
            from smartcash.ui.core.ui_module import SharedMethodRegistry
            
            # Register backbone operations
            SharedMethodRegistry.register_method(
                'backbone.execute_validate',
                self.execute_validate,
                description='Validate backbone configuration'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.execute_build', 
                self.execute_build,
                description='Build backbone model'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.execute_load',
                self.execute_load,
                description='Load pretrained backbone'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.execute_summary',
                self.execute_summary,
                description='Generate model summary'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.get_config',
                self.get_config,
                description='Get backbone configuration'
            )
            
            SharedMethodRegistry.register_method(
                'backbone.update_config',
                self.update_config,
                description='Update backbone configuration'
            )
            
            self.logger.debug("✅ Shared methods registered")
            
        except Exception as e:
            self.logger.warning(f"Failed to register shared methods: {e}")
    
    # ==================== OPERATION METHODS ====================
    
    def execute_validate(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone validation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Validation result dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_validate(config)
            
        except Exception as e:
            error_msg = f"Validation execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_build(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone build operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Build result dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_build(config)
            
        except Exception as e:
            error_msg = f"Build execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_load(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone load operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Load result dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_load(config)
            
        except Exception as e:
            error_msg = f"Load execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_summary(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute model summary generation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Summary result dictionary
        """
        try:
            if not self.is_initialized():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_summary(config)
            
        except Exception as e:
            error_msg = f"Summary execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    # ==================== STATUS AND INFO METHODS ====================
    
    def get_backbone_status(self) -> Dict[str, Any]:
        """
        Get current backbone module status.
        
        Returns:
            Status information dictionary
        """
        try:
            base_status = {
                'initialized': self.is_initialized(),
                'module_name': self.module_name,
                'parent_module': self.parent_module,
                'config_available': self._config_handler is not None,
                'operations_available': self._operation_manager is not None
            }
            
            if self._operation_manager:
                operation_status = self._operation_manager.get_status()
                base_status.update(operation_status)
            
            return base_status
            
        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return {'initialized': False, 'error': str(e)}
    
    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            UI components dictionary
        """
        return self._ui_components or {}
        
    def is_initialized(self) -> bool:
        """
        Check if the module is initialized.
        
        Returns:
            bool: True if the module is initialized and ready
        """
        return self._status == ModuleStatus.READY
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save current configuration.
        
        Returns:
            Save operation result
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Sync config from UI if available
            if self._ui_components:
                ui_config = self._config_handler.sync_from_ui(self._ui_components)
                if ui_config:
                    self.update_config(ui_config)
            
            # Save configuration (implementation depends on storage strategy)
            current_config = self.get_config()
            
            self.logger.info("📋 Configuration saved successfully")
            return {
                'success': True,
                'message': 'Configuration saved successfully',
                'config': current_config
            }
            
        except Exception as e:
            error_msg = f"Failed to save config: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Reset operation result
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Reset to default configuration
            default_config = get_default_backbone_config()
            self.update_config(default_config)
            
            # Sync to UI if available
            if self._ui_components:
                self._config_handler.sync_to_ui(self._ui_components, default_config)
            
            self.logger.info("🔄 Configuration reset to defaults")
            return {
                'success': True,
                'message': 'Configuration reset to defaults',
                'config': default_config
            }
            
        except Exception as e:
            error_msg = f"Failed to reset config: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        try:
            if self._operation_manager:
                self._operation_manager.cleanup()
            
            # Clear references
            self._config_handler = None
            self._operation_manager = None
            self._ui_components = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_backbone_uimodule_instance: Optional[BackboneUIModule] = None


def create_backbone_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> BackboneUIModule:
    """
    Create a new backbone UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        BackboneUIModule instance
    """
    module = BackboneUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def get_backbone_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> BackboneUIModule:
    """
    Get or create backbone UIModule singleton instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize if not exists
        **kwargs: Additional arguments
        
    Returns:
        BackboneUIModule singleton instance
    """
    global _backbone_uimodule_instance
    
    if _backbone_uimodule_instance is None:
        _backbone_uimodule_instance = create_backbone_uimodule(
            config=config,
            auto_initialize=auto_initialize,
            **kwargs
        )
    
    return _backbone_uimodule_instance


def reset_backbone_uimodule() -> None:
    """Reset the backbone UIModule singleton instance."""
    global _backbone_uimodule_instance
    
    if _backbone_uimodule_instance:
        _backbone_uimodule_instance.cleanup()
        _backbone_uimodule_instance = None


# ==================== CONVENIENCE FUNCTIONS ====================

def initialize_backbone_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Initialize backbone UI with convenience wrapper.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately
        **kwargs: Additional arguments
        
    Returns:
        Initialization result dictionary with keys:
        - success: bool indicating if initialization was successful
        - module: reference to the module instance
        - ui_components: dictionary of UI components
        - status: current module status
    """
    try:
        module = get_backbone_uimodule(config=config, **kwargs)
        
        result = {
            'success': True,
            'module': module,
            'ui_components': module.get_ui_components(),
            'status': module.get_backbone_status()
        }
        
        # Display UI if requested and components are available
        if display and result['ui_components']:
            from IPython.display import display as ipython_display
            main_ui = result['ui_components'].get('main_container')
            if main_ui:
                ipython_display(main_ui)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'module': None,
            'ui_components': {},
            'status': {}
        }


def get_backbone_components() -> Dict[str, Any]:
    """
    Get backbone UI components from singleton instance.
    
    Returns:
        UI components dictionary
    """
    try:
        module = get_backbone_uimodule(auto_initialize=False)
        return module.get_ui_components()
    except:
        return {}


# ==================== TEMPLATE REGISTRATION ====================

def register_backbone_shared_methods() -> None:
    """Register backbone shared methods for cross-module access."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register module factory functions
        SharedMethodRegistry.register_method(
            'backbone.create_module',
            create_backbone_uimodule,
            description='Create backbone UIModule instance'
        )
        
        SharedMethodRegistry.register_method(
            'backbone.get_module',
            get_backbone_uimodule,
            description='Get backbone UIModule singleton'
        )
        
        SharedMethodRegistry.register_method(
            'backbone.reset_module',
            reset_backbone_uimodule,
            description='Reset backbone UIModule singleton'
        )
        
    except Exception as e:
        # Silently fail if shared methods not available
        pass


def register_backbone_template() -> None:
    """Register backbone module template."""
    try:
        from smartcash.ui.core.template_registry import register_template
        
        template_info = {
            'name': 'backbone',
            'title': '🧬 Backbone Models',
            'description': 'Backbone model configuration with early training pipeline',
            'category': 'model',
            'factory_function': create_backbone_uimodule,
            'config_function': get_default_backbone_config,
            'singleton_function': get_backbone_uimodule,
            'reset_function': reset_backbone_uimodule
        }
        
        register_template('backbone', template_info)
        
    except Exception as e:
        # Silently fail if template registry not available
        pass


# Auto-register shared methods and template
register_backbone_shared_methods()
register_backbone_template()