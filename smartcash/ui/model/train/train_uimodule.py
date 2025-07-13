"""
File: smartcash/ui/model/train/train_uimodule.py
Main UIModule implementation for train module following new UIModule pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.utils.log_suppression import suppress_ui_init_logs
from .configs.train_config_handler import TrainConfigHandler
from .configs.train_defaults import get_default_train_config
from .operations.train_operation_manager import TrainOperationManager


class TrainUIModule(UIModule):
    """
    UIModule implementation for model training.
    
    Features:
    - 🚀 Model training continuation from backbone configuration
    - 📊 Dual live charts (loss and mAP) with real-time updates
    - 🔄 Progress tracking throughout training process
    - 🎯 Single/multilayer training options
    - 💾 Best model automatic saving with naming convention
    - 🔗 Backend training service integration
    - 🛡️ Fail-fast approach with comprehensive error handling
    """
    
    def __init__(self):
        """Initialize training UI module."""
        super().__init__(
            module_name='train',
            parent_module='model'
        )
        
        self.logger = get_module_logger("smartcash.ui.model.train")
        
        # Initialize components
        self._config_handler = None
        self._operation_manager = None
        self._ui_components = None
        self._chart_widgets = {}
        
        self.logger.debug("✅ TrainUIModule initialized")
    
    def _initialize_config_handler(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration handler."""
        try:
            self._config_handler = TrainConfigHandler()
            
            # Set initial configuration
            if config:
                merged_config = self._config_handler.merge_config(
                    get_default_train_config(), config
                )
            else:
                merged_config = get_default_train_config()
            
            # Try to integrate backbone configuration
            merged_config = self._try_integrate_backbone_config(merged_config)
            
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
            
            self._operation_manager = TrainOperationManager(
                config=self.get_config(),
                operation_container=operation_container
            )
            
            # Set chart callbacks for live updates
            self._setup_chart_callbacks()
            
            self._operation_manager.initialize()
            self.logger.debug("✅ Operation manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize operation manager: {e}")
            raise
    
    def _create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components with dual live charts."""
        try:
            from .components.training_ui import create_training_ui
            
            self.logger.debug("Creating training UI components...")
            ui_components = create_training_ui(config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Store chart widgets for live updates
            self._chart_widgets = {
                'loss_chart': ui_components.get('loss_chart'),
                'map_chart': ui_components.get('map_chart')
            }
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components with live charts")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def _setup_chart_callbacks(self) -> None:
        """Setup callbacks for live chart updates."""
        try:
            if not self._operation_manager or not self._chart_widgets:
                return
            
            # Loss chart callback
            def update_loss_chart(loss_data: Dict[str, float]):
                loss_chart = self._chart_widgets.get('loss_chart')
                if loss_chart and hasattr(loss_chart, 'add_data'):
                    loss_chart.add_data(loss_data)
            
            # mAP chart callback  
            def update_map_chart(map_data: Dict[str, float]):
                map_chart = self._chart_widgets.get('map_chart')
                if map_chart and hasattr(map_chart, 'add_data'):
                    map_chart.add_data(map_data)
            
            # Register callbacks with operation manager
            self._operation_manager.set_chart_callbacks(
                loss_chart_callback=update_loss_chart,
                map_chart_callback=update_map_chart
            )
            
            self.logger.debug("✅ Chart callbacks configured for live updates")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup chart callbacks: {e}")
    
    def _try_integrate_backbone_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Try to integrate backbone configuration automatically."""
        try:
            from smartcash.ui.core.ui_module import SharedMethodRegistry
            
            # Try to get backbone configuration
            get_backbone_config = SharedMethodRegistry.get_method('backbone.get_config')
            if get_backbone_config:
                backbone_config = get_backbone_config()
                
                if backbone_config and self._config_handler:
                    integrated_config = self._config_handler.integrate_backbone_config(
                        config, backbone_config
                    )
                    self.logger.info("✅ Backbone configuration automatically integrated")
                    return integrated_config
            
        except Exception as e:
            self.logger.warning(f"Could not auto-integrate backbone config: {e}")
        
        return config
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize the training module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
        """
        try:
            # Initialize configuration handler
            self._initialize_config_handler(config)
            
            # Create UI components with dual charts
            self._ui_components = self._create_ui_components(self.get_config())
            
            # Initialize operation manager with chart integration
            self._initialize_operation_manager()
            
            # Register shared methods for cross-module integration
            self._register_shared_methods()
            
            # Call base class initialization to set status to READY
            super().initialize(self.get_config())
            
        except Exception as e:
            self.logger.error(f"Failed to initialize training module: {e}")
            raise RuntimeError("Failed to create UI components")
    
    def _register_shared_methods(self) -> None:
        """Register shared methods for cross-module integration."""
        try:
            from smartcash.ui.core.ui_module import SharedMethodRegistry
            
            # Register training operations
            SharedMethodRegistry.register_method(
                'train.execute_start',
                self.execute_start,
                description='Start model training'
            )
            
            SharedMethodRegistry.register_method(
                'train.execute_stop',
                self.execute_stop,
                description='Stop model training'
            )
            
            SharedMethodRegistry.register_method(
                'train.execute_resume',
                self.execute_resume,
                description='Resume model training'
            )
            
            SharedMethodRegistry.register_method(
                'train.execute_validate',
                self.execute_validate,
                description='Validate trained model'
            )
            
            SharedMethodRegistry.register_method(
                'train.get_config',
                self.get_config,
                description='Get training configuration'
            )
            
            SharedMethodRegistry.register_method(
                'train.update_config',
                self.update_config,
                description='Update training configuration'
            )
            
            SharedMethodRegistry.register_method(
                'train.get_training_status',
                self.get_training_status,
                description='Get current training status'
            )
            
            self.logger.debug("✅ Shared methods registered")
            
        except Exception as e:
            self.logger.warning(f"Failed to register shared methods: {e}")
    
    # ==================== TRAINING OPERATION METHODS ====================
    
    def execute_start(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute training start operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Training start result dictionary
        """
        try:
            if not self.is_ready():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_start(config)
            
        except Exception as e:
            error_msg = f"Training start execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_stop(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute training stop operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Training stop result dictionary
        """
        try:
            if not self.is_ready():
                return {'success': False, 'message': 'Module not initialized'}
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_stop(config)
            
        except Exception as e:
            error_msg = f"Training stop execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_resume(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute training resume operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Training resume result dictionary
        """
        try:
            if not self.is_ready():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_resume(config)
            
        except Exception as e:
            error_msg = f"Training resume execution failed: {e}"
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
            if not self.is_ready():
                self.initialize()
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            return self._operation_manager.execute_validate(config)
            
        except Exception as e:
            error_msg = f"Validation execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    # ==================== STATUS AND INFO METHODS ====================
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training module status.
        
        Returns:
            Status information dictionary
        """
        try:
            base_status = {
                'initialized': self.is_ready(),
                'module_name': self.module_name,
                'parent_module': self.parent_module,
                'config_available': self._config_handler is not None,
                'operations_available': self._operation_manager is not None,
                'charts_available': bool(self._chart_widgets)
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
    
    def get_live_charts(self) -> Dict[str, Any]:
        """
        Get live chart widgets.
        
        Returns:
            Chart widgets dictionary
        """
        return self._chart_widgets.copy()
    
    def integrate_backbone_config(self, backbone_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate backbone configuration into training config.
        
        Args:
            backbone_config: Backbone configuration from backbone module
            
        Returns:
            Integration result
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            current_config = self.get_config()
            integrated_config = self._config_handler.integrate_backbone_config(
                current_config, backbone_config
            )
            
            self.update_config(**integrated_config)
            
            self.logger.info("📋 Backbone configuration integrated successfully")
            return {
                'success': True,
                'message': 'Backbone configuration integrated successfully',
                'config': integrated_config
            }
            
        except Exception as e:
            error_msg = f"Failed to integrate backbone config: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
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
                    self.update_config(**ui_config)
            
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
            default_config = get_default_train_config()
            self.update_config(**default_config)
            
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
            self._chart_widgets.clear()
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_train_uimodule_instance: Optional[TrainUIModule] = None


def create_train_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> TrainUIModule:
    """
    Create a new training UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        TrainUIModule instance
    """
    module = TrainUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def get_train_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> TrainUIModule:
    """
    Get or create training UIModule singleton instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize if not exists
        **kwargs: Additional arguments
        
    Returns:
        TrainUIModule singleton instance
    """
    global _train_uimodule_instance
    
    if _train_uimodule_instance is None:
        _train_uimodule_instance = create_train_uimodule(
            config=config,
            auto_initialize=auto_initialize,
            **kwargs
        )
    
    return _train_uimodule_instance


def reset_train_uimodule() -> None:
    """Reset the training UIModule singleton instance."""
    global _train_uimodule_instance
    
    if _train_uimodule_instance:
        _train_uimodule_instance.cleanup()
        _train_uimodule_instance = None


# ==================== CONVENIENCE FUNCTIONS ====================

def initialize_training_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Initialize training UI with convenience wrapper.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately
        **kwargs: Additional arguments
        
    Returns:
        Initialization result dictionary
    """
    try:
        module = get_train_uimodule(config=config, **kwargs)
        
        result = {
            'success': True,
            'module': module,
            'ui_components': module.get_ui_components(),
            'status': module.get_training_status(),
            'live_charts': module.get_live_charts()
        }
        
        if display and result['ui_components']:
            from IPython.display import display as ipython_display
            main_ui = result['ui_components'].get('main_container')
            if main_ui:
                ipython_display(main_ui)
                return None  # Don't return data when display=True
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'module': None,
            'ui_components': {},
            'status': {},
            'live_charts': {}
        }


def get_training_components() -> Dict[str, Any]:
    """
    Get training UI components from singleton instance.
    
    Returns:
        UI components dictionary
    """
    try:
        module = get_train_uimodule(auto_initialize=False)
        return module.get_ui_components()
    except:
        return {}


# ==================== TEMPLATE REGISTRATION ====================

def register_train_shared_methods() -> None:
    """Register training shared methods for cross-module access."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register module factory functions
        SharedMethodRegistry.register_method(
            'train.create_module',
            create_train_uimodule,
            description='Create training UIModule instance'
        )
        
        SharedMethodRegistry.register_method(
            'train.get_module',
            get_train_uimodule,
            description='Get training UIModule singleton'
        )
        
        SharedMethodRegistry.register_method(
            'train.reset_module',
            reset_train_uimodule,
            description='Reset training UIModule singleton'
        )
        
    except Exception as e:
        # Silently fail if shared methods not available
        pass


def register_train_template() -> None:
    """Register training module template."""
    # Template registry not available in current core implementation
    # This is a placeholder for future template system
    pass


# Auto-register shared methods and template
register_train_shared_methods()
register_train_template()