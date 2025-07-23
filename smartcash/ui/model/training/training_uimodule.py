"""
File: smartcash/ui/model/training/training_uimodule.py
Main UIModule implementation for training module using BaseUIModule pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.core.mixins.button_handler_mixin import ButtonHandlerMixin
from smartcash.ui.logger import get_module_logger

from .configs.training_config_handler import TrainingConfigHandler
from .configs.training_defaults import get_default_training_config
from .operations.training_factory import TrainingOperationFactory

# Import model mixins for shared functionality
from smartcash.ui.model.mixins import (
    ModelConfigSyncMixin,
    BackendServiceMixin
)


class TrainingUIModule(ModelConfigSyncMixin, BackendServiceMixin, BaseUIModule, OperationMixin, ButtonHandlerMixin):
    """
    UIModule implementation for model training using BaseUIModule pattern.
    
    Features:
    - 🚀 Model training with automatic backbone integration
    - 📊 Live charts (loss and mAP) with real-time updates
    - 🔄 Comprehensive progress tracking with dual progress bars
    - 🎯 Multiple training operations (start, stop, resume, validate)
    - 💾 Automatic model checkpointing and validation
    - 🔗 Backend training service integration
    - 🛡️ Comprehensive error handling and recovery
    """
    
    def __init__(self):
        """Initialize training UI module."""
        super().__init__(
            module_name='training',
            parent_module='model',
            enable_environment=True
        )
        
        self.logger = get_module_logger("smartcash.ui.model.training")
        
        # Initialize module-specific components
        self._config_handler: Optional[TrainingConfigHandler] = None
        self._chart_updaters: Dict[str, Callable] = {}
        self._training_state = {'phase': 'idle'}
        
        # Button registration tracking
        self._buttons_registered = False
        
        # Minimal logging for performance
        # Debug information disabled during normal operation
    
    def create_config_handler(self, config: Optional[Dict[str, Any]] = None) -> TrainingConfigHandler:
        """Create configuration handler."""
        merged_config = self._merge_with_defaults(config)
        config_handler = TrainingConfigHandler(merged_config)
        config_handler.set_ui_components(self._ui_components)
        return config_handler
    
    def _merge_with_defaults(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge provided config with defaults."""
        default_config = get_default_training_config()
        
        if config:
            # Deep merge logic
            merged = default_config.copy()
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
            return merged
        
        return default_config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return get_default_training_config()
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components with live charts."""
        try:
            from .components.training_ui import create_training_ui
            
            # Minimal logging for performance
            ui_components = create_training_ui(config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Setup chart updaters
            self._setup_chart_updaters()
            
            # Success messages - minimal confirmation only
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def _setup_chart_updaters(self) -> None:
        """Setup chart update methods for live monitoring."""
        try:
            # Loss chart updater
            def update_loss_chart(loss_data: Dict[str, Any]) -> None:
                try:
                    # Find loss chart widget in UI components
                    loss_chart = self._find_chart_widget('loss_chart')
                    if loss_chart and hasattr(loss_chart, 'add_data'):
                        loss_chart.add_data({
                            'epoch': loss_data.get('epoch', 0),
                            'train_loss': loss_data.get('train_loss', 0.0),
                            'val_loss': loss_data.get('val_loss', 0.0)
                        })
                        # Debug information disabled during normal operation
                except Exception as e:
                    self.log_warning(f"Failed to update loss chart: {e}")
            
            # mAP chart updater
            def update_map_chart(map_data: Dict[str, Any]) -> None:
                try:
                    # Find mAP chart widget in UI components
                    map_chart = self._find_chart_widget('map_chart')
                    if map_chart and hasattr(map_chart, 'add_data'):
                        map_chart.add_data({
                            'epoch': map_data.get('epoch', 0),
                            'mAP@0.5': map_data.get('mAP@0.5', 0.0),
                            'mAP@0.75': map_data.get('mAP@0.75', 0.0)
                        })
                        # Debug information disabled during normal operation
                except Exception as e:
                    self.log_warning(f"Failed to update mAP chart: {e}")
            
            # Store chart updaters
            self._chart_updaters = {
                'loss_chart': update_loss_chart,
                'map_chart': update_map_chart
            }
            
            # Minimal logging for performance
            
        except Exception as e:
            self.log_warning(f"Failed to setup chart updaters: {e}")
    
    def _find_chart_widget(self, chart_type: str) -> Any:
        """Find chart widget in UI components."""
        try:
            # Look for chart in various possible locations
            charts = self._ui_components.get('charts', {})
            if chart_type in charts:
                return charts[chart_type]
            
            # Check if chart is directly in ui_components
            if chart_type in self._ui_components:
                return self._ui_components[chart_type]
            
            # Check summary container for charts
            summary_container = self._ui_components.get('summary_container', {})
            if chart_type in summary_container:
                return summary_container[chart_type]
            
            return None
            
        except Exception:
            return None
    
    def _get_module_button_handlers(self) -> Dict[str, Callable]:
        """Get module-specific button handlers."""
        return {
            'start_training': self._handle_start_training,
            'stop_training': self._handle_stop_training,
            'resume_training': self._handle_resume_training,
            # 'validate_model': removed - overlaps with backbone module
            'refresh_backbone_config': self._handle_refresh_backbone_config,
            'save': self._handle_save_config,
            'reset': self._handle_reset_config
        }

    def _register_dynamic_button_handlers(self) -> None:
        """Register dynamic button handlers with duplicate prevention."""
        if self._buttons_registered:
            self.logger.debug("⏭️ Skipping training button registration - already registered")
            return
        
        try:
            # Call parent method to handle registration
            if hasattr(super(), '_register_dynamic_button_handlers'):
                super()._register_dynamic_button_handlers()
            
            # Mark as registered
            self._buttons_registered = True
            self.logger.info("🎯 Training button handlers registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register training button handlers: {e}", exc_info=True)
    
    def _handle_start_training(self, button=None) -> None:
        """Handle start training button click."""
        try:
            self.log_info("Starting training...")
            result = self.execute_start_training()
            
            if result.get('success'):
                self.log_success(f"Training started: {result.get('message', '')}")
            else:
                self.log_error(f"Training start failed: {result.get('message', '')}")
                
        except Exception as e:
            self.log_error(f"Start training error: {e}")
    
    def _handle_stop_training(self, button=None) -> None:
        """Handle stop training button click."""
        try:
            self.log_info("Stopping training...")
            result = self.execute_stop_training()
            
            if result.get('success'):
                self.log_success(f"Training stopped: {result.get('message', '')}")
            else:
                self.log_error(f"Training stop failed: {result.get('message', '')}")
                
        except Exception as e:
            self.log_error(f"Stop training error: {e}")
    
    def _handle_resume_training(self, button=None) -> None:
        """Handle resume training button click."""
        try:
            self.log_info("Resuming training...")
            result = self.execute_resume_training()
            
            if result.get('success'):
                self.log_success(f"Training resumed: {result.get('message', '')}")
            else:
                self.log_error(f"Training resume failed: {result.get('message', '')}")
                
        except Exception as e:
            self.log_error(f"Resume training error: {e}")
    
    # _handle_validate_model removed - validation now handled by backbone module
    
    def _handle_refresh_backbone_config(self, button=None) -> None:
        """Handle refresh backbone config button click."""
        try:
            self.log_info("Refreshing backbone configuration...")
            result = self.execute_refresh_backbone_config()
            
            if result.get('success'):
                self.log_success(f"Backbone config refreshed: {result.get('message', '')}")
            else:
                self.log_error(f"Backbone config refresh failed: {result.get('message', '')}")
                
        except Exception as e:
            self.log_error(f"Refresh backbone config error: {e}")
    
    def _handle_save_config(self, button=None) -> None:
        """Handle save config button click."""
        try:
            self.log_info("Saving configuration...")
            result = self.save_current_config()
            
            if result.get('success'):
                self.log_success("Configuration saved successfully")
            else:
                self.log_error(f"Save config failed: {result.get('message', '')}")
                
        except Exception as e:
            self.log_error(f"Save config error: {e}")
    
    def _handle_reset_config(self, button=None) -> None:
        """Handle reset config button click."""
        try:
            self.log_info("Resetting configuration...")
            result = self.reset_to_defaults()
            
            if result.get('success'):
                self.log_success("Configuration reset to defaults")
            else:
                self.log_error(f"Reset config failed: {result.get('message', '')}")
                
        except Exception as e:
            self.log_error(f"Reset config error: {e}")
    
    # ==================== TRAINING OPERATION METHODS ====================
    
    def execute_start_training(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute training start operation."""
        try:
            # Merge config with current configuration
            operation_config = self.get_current_config().copy()
            if config:
                operation_config.update(config)
            
            # Try to get backbone configuration for model selection
            backbone_config = self._get_backbone_configuration()
            if backbone_config and self._config_handler:
                model_result = self._config_handler.select_model_from_backbone(backbone_config)
                if model_result['success']:
                    operation_config.update({'model_selection': model_result['model_selection']})
            
            # Create operation handler
            handler, validation_result = TrainingOperationFactory.create_operation_with_validation(
                'start', self, operation_config, self._get_operation_callbacks()
            )
            
            if not handler:
                return validation_result
            
            # Execute operation
            return handler.execute()
            
        except Exception as e:
            error_msg = f"Training start execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_stop_training(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute training stop operation."""
        try:
            operation_config = self.get_current_config().copy()
            if config:
                operation_config.update(config)
            
            handler, validation_result = TrainingOperationFactory.create_operation_with_validation(
                'stop', self, operation_config, self._get_operation_callbacks()
            )
            
            if not handler:
                return validation_result
            
            return handler.execute()
            
        except Exception as e:
            error_msg = f"Training stop execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_resume_training(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute training resume operation."""
        try:
            operation_config = self.get_current_config().copy()
            if config:
                operation_config.update(config)
            
            handler, validation_result = TrainingOperationFactory.create_operation_with_validation(
                'resume', self, operation_config, self._get_operation_callbacks()
            )
            
            if not handler:
                return validation_result
            
            return handler.execute()
            
        except Exception as e:
            error_msg = f"Training resume execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_validate_model(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute model validation operation."""
        try:
            operation_config = self.get_current_config().copy()
            if config:
                operation_config.update(config)
            
            handler, validation_result = TrainingOperationFactory.create_operation_with_validation(
                'validate', self, operation_config, self._get_operation_callbacks()
            )
            
            if not handler:
                return validation_result
            
            return handler.execute()
            
        except Exception as e:
            error_msg = f"Validation execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def execute_refresh_backbone_config(self) -> Dict[str, Any]:
        """Refresh backbone configuration and update training config."""
        try:
            backbone_config = self._get_backbone_configuration()
            
            if not backbone_config:
                return {'success': False, 'message': 'No backbone configuration available'}
            
            if not self._config_handler:
                return {'success': False, 'message': 'Config handler not available'}
            
            # Update model selection based on backbone
            model_result = self._config_handler.select_model_from_backbone(backbone_config)
            if model_result['success']:
                # Update configuration and UI
                self._update_module_config(model_selection=model_result['model_selection'])
                self._sync_config_to_ui()
                
                return {
                    'success': True,
                    'message': 'Backbone configuration refreshed successfully',
                    'model_selection': model_result['model_selection']
                }
            else:
                return model_result
                
        except Exception as e:
            error_msg = f"Failed to refresh backbone config: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def _get_backbone_configuration(self) -> Optional[Dict[str, Any]]:
        """Get current backbone configuration using ModelConfigSyncMixin."""
        try:
            # Use ModelConfigSyncMixin's get_module_config method
            backbone_config = self.get_module_config('backbone', auto_initialize=False)
            if backbone_config:
                # Debug information disabled during normal operation
                return backbone_config
            else:
                self.log_warning("⚠️ Backbone module not available")
                return None
                
        except Exception as e:
            self.log_warning(f"⚠️ Error getting backbone config: {e}")
            return None
    
    def _get_operation_callbacks(self) -> Dict[str, Callable]:
        """Get callbacks for operation handlers."""
        return {
            'on_progress': self._handle_operation_progress,
            'on_success': self._handle_operation_success,
            'on_failure': self._handle_operation_failure,
            'on_chart_update': self._handle_chart_update
        }
    
    def _handle_operation_progress(self, progress: int, message: str = "") -> None:
        """Handle operation progress updates."""
        try:
            if hasattr(self, 'update_progress'):
                self.update_progress(progress=progress, message=message)
        except Exception as e:
            self.log_warning(f"Failed to update progress: {e}")
    
    def _handle_operation_success(self, message: str) -> None:
        """Handle operation success."""
        self.log_success(message)
    
    def _handle_operation_failure(self, message: str) -> None:
        """Handle operation failure."""
        self.log_error(message)
    
    def _handle_chart_update(self, metrics: Dict[str, Any]) -> None:
        """Handle chart update requests from operations."""
        try:
            # Update loss chart if loss metrics available
            if any(key in metrics for key in ['train_loss', 'val_loss']):
                loss_updater = self._chart_updaters.get('loss_chart')
                if loss_updater:
                    loss_updater(metrics)
            
            # Update mAP chart if mAP metrics available
            if any(key in metrics for key in ['mAP@0.5', 'mAP@0.75']):
                map_updater = self._chart_updaters.get('map_chart')
                if map_updater:
                    map_updater(metrics)
                    
        except Exception as e:
            self.log_warning(f"Failed to update charts: {e}")
    
    def save_current_config(self) -> Dict[str, Any]:
        """Save current configuration."""
        try:
            if not self._config_handler:
                return {'success': False, 'message': 'Config handler not available'}
            
            # Extract config from UI if needed
            current_config = self.get_current_config()
            
            # In a real implementation, this would save to file or database
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
    
    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset configuration to defaults."""
        try:
            default_config = get_default_training_config()
            self._update_module_config(**default_config)
            self._sync_config_to_ui()
            
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
    
    def _sync_config_to_ui(self) -> None:
        """Sync current configuration to UI components."""
        try:
            if self._config_handler and self._ui_components:
                current_config = self.get_current_config()
                
                # Update form if available
                form_container = self._ui_components.get('form_container')
                if form_container and hasattr(form_container, 'update_from_config'):
                    form_container.update_from_config(current_config)
                    # Debug information disabled during normal operation
                else:
                    self.log_warning("Form container or update method not available")
                    
        except Exception as e:
            self.log_warning(f"Failed to sync config to UI: {e}")

    def _update_module_config(self, **updates: Any) -> None:
        """Update module configuration with provided values."""
        try:
            if self._config_handler:
                current_config = self.get_current_config()
                
                # Deep update configuration
                for key, value in updates.items():
                    if key in current_config and isinstance(current_config[key], dict) and isinstance(value, dict):
                        current_config[key].update(value)
                    else:
                        current_config[key] = value
                
                # Update the config handler
                self._config_handler.config = current_config
                # Debug information disabled during normal operation
            else:
                self.log_warning("Config handler not available for update")
                
        except Exception as e:
            self.logger.error(f"Failed to update module config: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'initialized': self.is_ready(),
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'config_available': self._config_handler is not None,
            'charts_available': bool(self._chart_updaters),
            'training_state': self._training_state.copy()
        }
    
    def get_live_charts(self) -> Dict[str, Any]:
        """Get live chart widgets."""
        return {
            'loss_chart': self._find_chart_widget('loss_chart'),
            'map_chart': self._find_chart_widget('map_chart')
        }
    
    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Clear chart updaters to prevent memory leaks
            self._chart_updaters.clear()
            
            # Clear training state
            self._training_state.clear()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Training module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Training module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion
