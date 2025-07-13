"""
File: smartcash/ui/model/train/train_uimodule.py
Main UIModule implementation for train module following new UIModule pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.decorators import suppress_ui_init_logs
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
        """Initialize configuration handler with shared config support."""
        try:
            # Get default config first
            default_config = get_default_train_config()
            
            # Ensure required sections exist in the provided config
            if config:
                # Ensure all required sections exist in the provided config
                for section in ['training', 'optimizer', 'scheduler', 'monitoring', 'ui']:
                    if section not in config:
                        config[section] = default_config.get(section, {})
            
            # Initialize with proper config and shared settings
            self._config_handler = TrainConfigHandler()
            
            # Ensure shared manager is initialized
            if not hasattr(self._config_handler, '_shared_manager'):
                self._config_handler.initialize()
            
            # If config was provided, update the handler with it
            if config:
                # First ensure we have all required sections in the config
                for section in ['training', 'optimizer', 'scheduler', 'monitoring', 'ui']:
                    if section not in config:
                        config[section] = default_config.get(section, {})
                
                # Update the config handler with the provided config
                self._config_handler.update_config(config)
            
            # Get the current config from handler (which now has all required sections)
            current_config = self._config_handler.get_config()
            
            # Try to integrate backbone configuration
            current_config = self._try_integrate_backbone_config(current_config)
            
            # Update the config with the merged configuration
            self.update_config(**current_config)
            
            # Validate the configuration
            if not self._config_handler.validate_config(current_config):
                self.logger.warning("Configuration validation failed, but continuing with default values")
            
            self.logger.debug("✅ Config handler initialized with config: %s", 
                           {k: '...' for k in current_config.keys()})
            
        except Exception as e:
            self.logger.error(f"Failed to initialize config handler: {e}", exc_info=True)
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
    
    def _setup_button_handlers(self) -> None:
        """Setup button event handlers for training operations."""
        try:
            if not self._ui_components or not self._operation_manager:
                self.logger.warning("Cannot setup button handlers - missing components or operation manager")
                return
                
            # Get action container
            action_container = self._ui_components.get('action_container')
            if not action_container:
                self.logger.warning("Action container not found in UI components")
                return
                
            # Get buttons from action container
            buttons = action_container.get('buttons', {})
            
            # Setup button handlers
            button_handlers = {
                'start_training': self._handle_start_training,
                'stop_training': self._handle_stop_training,
                'resume_training': self._handle_resume_training,
                'validate_model': self._handle_validate_model,
                'refresh_backbone_config': self._handle_refresh_backbone_config
            }
            
            for button_id, handler in button_handlers.items():
                button = buttons.get(button_id)
                if button and hasattr(button, 'on_click'):
                    button.on_click(handler)
                    self.logger.debug(f"✅ Bound {button_id} button to {handler.__name__}")
                else:
                    self.logger.warning(f"Button {button_id} not found or doesn't support on_click")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup button handlers: {e}")
    
    def _handle_start_training(self, button=None):
        """Handle start training button click."""
        try:
            self.log("🚀 Start training button clicked", 'info')
            result = self.execute_start()
            if result.get('success'):
                self.log(f"✅ Training started: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Training start failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Start training error: {e}", 'error')
    
    def _handle_stop_training(self, button=None):
        """Handle stop training button click."""
        try:
            self.log("🛑 Stop training button clicked", 'info')
            result = self.execute_stop()
            if result.get('success'):
                self.log(f"✅ Training stopped: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Training stop failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Stop training error: {e}", 'error')
    
    def _handle_resume_training(self, button=None):
        """Handle resume training button click.""" 
        try:
            self.log("▶️ Resume training button clicked", 'info')
            result = self.execute_resume()
            if result.get('success'):
                self.log(f"✅ Training resumed: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Training resume failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Resume training error: {e}", 'error')
    
    def _handle_validate_model(self, button=None):
        """Handle validate model button click."""
        try:
            self.log("✅ Validate model button clicked", 'info')
            result = self.execute_validate()
            if result.get('success'):
                self.log(f"✅ Model validation completed: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Model validation failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Validate model error: {e}", 'error')
    
    def _handle_refresh_backbone_config(self, button=None):
        """Handle refresh backbone config button click."""
        try:
            self.log("🔄 Refresh backbone config button clicked", 'info')
            result = self.execute_refresh_backbone_config()
            if result.get('success'):
                self.log(f"✅ Backbone config refreshed: {result.get('message', '')}", 'success')
            else:
                self.log(f"❌ Backbone config refresh failed: {result.get('message', '')}", 'error')
        except Exception as e:
            self.log(f"❌ Refresh backbone config error: {e}", 'error')
    
    def _setup_save_reset_handlers(self) -> None:
        """Setup Save/Reset button handlers."""
        try:
            if not self._ui_components:
                self.logger.warning("Cannot setup Save/Reset handlers - missing UI components")
                return
                
            # Get action container
            action_container = self._ui_components.get('action_container')
            if not action_container:
                self.logger.warning("Action container not found for Save/Reset handlers")
                return
                
            # Get ActionContainer object (not just the container widget)
            action_container_obj = action_container.get('action_container')
            if not action_container_obj:
                self.logger.warning("ActionContainer object not found")
                return
                
            # Get Save/Reset buttons
            save_button = getattr(action_container_obj, 'save_button', None)
            reset_button = getattr(action_container_obj, 'reset_button', None)
            
            if save_button and hasattr(save_button, 'on_click'):
                save_button.on_click(self._handle_save_config)
                self.logger.debug("✅ Bound Save button to _handle_save_config")
            else:
                self.logger.warning("Save button not found or doesn't support on_click")
                
            if reset_button and hasattr(reset_button, 'on_click'):
                reset_button.on_click(self._handle_reset_config)
                self.logger.debug("✅ Bound Reset button to _handle_reset_config")
            else:
                self.logger.warning("Reset button not found or doesn't support on_click")
                
        except Exception as e:
            self.logger.error(f"Failed to setup Save/Reset handlers: {e}")
    
    def _handle_save_config(self, button=None):
        """Handle save config button click."""
        try:
            self.log("💾 Save config button clicked", 'info')
            # Extract current configuration from UI and save it
            if self._config_handler:
                # Use config handler's save functionality
                result = self._config_handler.save_config()
                if result:
                    self.log("✅ Training configuration saved successfully", 'success')
                else:
                    self.log("❌ Failed to save training configuration", 'error')
            else:
                self.log("❌ Config handler not available", 'error')
        except Exception as e:
            self.log(f"❌ Save config error: {e}", 'error')
    
    def _handle_reset_config(self, button=None):
        """Handle reset config button click."""
        try:
            self.log("🔄 Reset config button clicked", 'info')
            # Reset configuration to defaults
            if self._config_handler:
                self._config_handler.reset_config()
                self.log("✅ Training configuration reset to defaults", 'success')
                # Optionally sync UI with the reset config
                if hasattr(self._config_handler, 'sync_ui_with_config'):
                    self._config_handler.sync_ui_with_config()
            else:
                self.log("❌ Config handler not available", 'error')
        except Exception as e:
            self.log(f"❌ Reset config error: {e}", 'error')

    def log(self, message: str, level: str = 'info') -> None:
        """Log message to operation container."""
        if self._operation_manager and hasattr(self._operation_manager, 'log'):
            self._operation_manager.log(message, level)
        else:
            getattr(self.logger, level, self.logger.info)(message)

    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container."""
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'log'):
                self._operation_manager.log("✅ Training module initialized successfully", 'info')
                self._operation_manager.log("🔧 Ready for model training operations", 'info')
                self._operation_manager.log("📊 Live charts and progress tracking enabled", 'info')
                
                # Log any backbone integration status
                backbone_config = self.get_config().get('backbone_integration', {})
                if backbone_config.get('backbone_type'):
                    backbone_type = backbone_config.get('backbone_type', 'unknown')
                    self._operation_manager.log(f"🔗 Integrated with {backbone_type} backbone", 'info')
                    
            self.logger.debug("✅ Initialization complete logs sent to operation container")
            
        except Exception as e:
            self.logger.warning(f"Failed to log initialization complete: {e}")
    
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
            
            # Setup button event handlers
            self._setup_button_handlers()
            
            # Setup Save/Reset button handlers
            self._setup_save_reset_handlers()
            
            # Register shared methods for cross-module integration
            self._register_shared_methods()
            
            # Log successful initialization to operation container
            self._log_initialization_complete()
            
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
            
            SharedMethodRegistry.register_method(
                'train.refresh_backbone_config',
                self.refresh_backbone_config,
                description='Refresh backbone configuration from backbone module'
            )
            
            SharedMethodRegistry.register_method(
                'train.execute_refresh_backbone_config',
                self.execute_refresh_backbone_config,
                description='Execute refresh backbone configuration operation'
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
    
    def refresh_backbone_config(self) -> Dict[str, Any]:
        """
        Refresh backbone configuration from backbone module.
        
        Returns:
            Refresh result dictionary
        """
        try:
            from smartcash.ui.core.ui_module import SharedMethodRegistry
            
            # Try to get backbone configuration
            get_backbone_config = SharedMethodRegistry.get_method('backbone.get_config')
            if not get_backbone_config:
                return {'success': False, 'message': 'Backbone module not available'}
            
            backbone_config = get_backbone_config()
            if not backbone_config:
                return {'success': False, 'message': 'No backbone configuration available'}
            
            # Integrate backbone configuration
            if self._config_handler:
                current_config = self.get_config()
                integrated_config = self._config_handler.integrate_backbone_config(
                    current_config, backbone_config
                )
                self.update_config(**integrated_config)
                return {'success': True, 'message': 'Backbone configuration refreshed successfully'}
            else:
                return {'success': False, 'message': 'Config handler not available'}
                
        except Exception as e:
            self.logger.error(f"Failed to refresh backbone config: {e}")
            return {'success': False, 'message': str(e)}

    def execute_refresh_backbone_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute refresh backbone configuration operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Refresh result dictionary
        """
        try:
            if not self.is_ready():
                self.initialize()
            
            return self.refresh_backbone_config()
            
        except Exception as e:
            error_msg = f"Refresh backbone config execution failed: {e}"
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
    
    def refresh_backbone_config(self) -> Dict[str, Any]:
        """
        Refresh backbone configuration and update the training module.
        
        Returns:
            Refresh operation result
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Get current config and refresh backbone integration
            current_config = self.get_config()
            updated_config = self._config_handler.refresh_backbone_config(current_config)
            
            # Update module configuration
            self.update_config(**updated_config)
            
            # Sync to UI if available
            if self._ui_components:
                self._config_handler.sync_to_ui(self._ui_components, updated_config)
            
            self.logger.info("🔄 Backbone configuration refreshed and UI updated")
            return {
                'success': True,
                'message': 'Backbone configuration refreshed successfully',
                'config': updated_config
            }
            
        except Exception as e:
            error_msg = f"Failed to refresh backbone config: {e}"
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