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
        
        # Training-specific state for button dependencies
        self._has_model = False
        self._has_checkpoint = False
        self._is_training_active = False
        
        # Setup training-specific button dependencies
        self._setup_training_button_dependencies()
        
        # Initialize training state from available data
        self._initialize_training_state()
        
        # Minimal logging for performance
        # Debug information disabled during normal operation
    
    def create_config_handler(self, config: Optional[Dict[str, Any]] = None) -> TrainingConfigHandler:
        """Create configuration handler with proper delegation."""
        # Use ConfigurationMixin's merge functionality instead of duplicating logic
        merged_config = config or get_default_training_config()
        config_handler = TrainingConfigHandler(merged_config)
        if hasattr(self, '_ui_components') and self._ui_components:
            config_handler.set_ui_components(self._ui_components)
        return config_handler
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration - delegates to defaults module."""
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
    
    def _setup_training_button_dependencies(self) -> None:
        """
        Setup button dependencies specific to training module requirements.
        
        This uses the enhanced button mixin functionality to define when
        training buttons should be enabled/disabled based on module state.
        """
        try:
            # Start training button depends on having a model selected
            self.set_button_dependency('start_training', self._check_start_training_dependency)
            
            # Resume training button depends on having a checkpoint
            self.set_button_dependency('resume_training', self._check_resume_training_dependency)
            
            # Stop training button depends on training being active
            self.set_button_dependency('stop_training', self._check_stop_training_dependency)
            
            # Validate model button depends on having model or checkpoint
            self.set_button_dependency('validate_model', self._check_validate_model_dependency)
            
            self.logger.debug("✅ Training button dependencies configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup training button dependencies: {e}")
    
    def _check_start_training_dependency(self) -> bool:
        """
        Check if start training button should be enabled.
        
        Returns:
            True if start training should be enabled (has model and not training)
        """
        return self._has_model and not self._is_training_active
    
    def _check_resume_training_dependency(self) -> bool:
        """
        Check if resume training button should be enabled.
        
        Returns:
            True if resume training should be enabled (has checkpoint and not training)
        """
        return self._has_checkpoint and not self._is_training_active
    
    def _check_stop_training_dependency(self) -> bool:
        """
        Check if stop training button should be enabled.
        
        Returns:
            True if stop training should be enabled (training is active)
        """
        return self._is_training_active
    
    def _check_validate_model_dependency(self) -> bool:
        """
        Check if validate model button should be enabled.
        
        Returns:
            True if validate model should be enabled (has model or checkpoint)
        """
        return self._has_model or self._has_checkpoint
    
    def _update_training_state(self, has_model: bool = None, has_checkpoint: bool = None, is_training_active: bool = None) -> None:
        """
        Update training state and refresh button states using enhanced mixin.
        
        Args:
            has_model: Whether a model is available
            has_checkpoint: Whether a checkpoint is available
            is_training_active: Whether training is currently active
        """
        try:
            # Update state variables
            if has_model is not None:
                self._has_model = has_model
            if has_checkpoint is not None:
                self._has_checkpoint = has_checkpoint
            if is_training_active is not None:
                self._is_training_active = is_training_active
            
            # Update button states using enhanced mixin functionality
            self._update_training_button_states()
            
        except Exception as e:
            self.logger.error(f"Failed to update training state: {e}")
    
    def _update_training_button_states(self) -> None:
        """
        Update all training button states based on current conditions.
        
        This replaces the custom button state logic with the enhanced mixin approach.
        """
        try:
            # Define button conditions based on current state
            button_conditions = {
                'start_training': self._has_model and not self._is_training_active,
                'resume_training': self._has_checkpoint and not self._is_training_active,
                'stop_training': self._is_training_active,
                'validate_model': self._has_model or self._has_checkpoint
            }
            
            # Define reasons for disabled buttons
            button_reasons = {
                'start_training': self._get_start_training_disable_reason(),
                'resume_training': self._get_resume_training_disable_reason(),
                'stop_training': "No active training to stop" if not self._is_training_active else None,
                'validate_model': "No model or checkpoint available" if not (self._has_model or self._has_checkpoint) else None
            }
            
            # Update all button states in one call using enhanced mixin
            self.update_button_states_based_on_condition(button_conditions, button_reasons)
            
            self.logger.debug("Training button states updated based on current conditions")
            
        except Exception as e:
            self.logger.error(f"Failed to update training button states: {e}")
    
    def _get_start_training_disable_reason(self) -> Optional[str]:
        """Get reason why start training button is disabled."""
        if not self._has_model:
            return "No model selected - configure backbone first"
        elif self._is_training_active:
            return "Training already in progress"
        return None
    
    def _get_resume_training_disable_reason(self) -> Optional[str]:
        """Get reason why resume training button is disabled."""
        if not self._has_checkpoint:
            return "No checkpoint available to resume from"
        elif self._is_training_active:
            return "Training already in progress"
        return None
    
    def _handle_start_training(self, button=None) -> None:
        """Handle start training button click."""
        try:
            self.log_info("Starting training...")
            
            # Update state to indicate training is starting
            self._update_training_state(is_training_active=True)
            
            result = self.execute_start_training()
            
            if result.get('success'):
                self.log_success(f"Training started: {result.get('message', '')}")
            else:
                # Reset training state if start failed
                self._update_training_state(is_training_active=False)
                self.log_error(f"Training start failed: {result.get('message', '')}")
                
        except Exception as e:
            # Reset training state on error
            self._update_training_state(is_training_active=False)
            self.log_error(f"Start training error: {e}")
    
    def _handle_stop_training(self, button=None) -> None:
        """Handle stop training button click."""
        try:
            self.log_info("Stopping training...")
            
            # Update state to indicate training is stopping
            self._update_training_state(is_training_active=False)
            
            result = self.execute_stop_training()
            
            if result.get('success'):
                self.log_success(f"Training stopped: {result.get('message', '')}")
                # Update state to reflect stopped training
                self._update_training_state(is_training_active=False)
            else:
                self.log_error(f"Training stop failed: {result.get('message', '')}")
                # Reset training state if stop failed
                self._update_training_state(is_training_active=True)
                
        except Exception as e:
            # Reset training state on error
            self._update_training_state(is_training_active=True)
            self.log_error(f"Stop training error: {e}")
    
    def _handle_resume_training(self, button=None) -> None:
        """Handle resume training button click."""
        try:
            self.log_info("Resuming training...")
            
            # Update state to indicate training is resuming
            self._update_training_state(is_training_active=True)
            
            result = self.execute_resume_training()
            
            if result.get('success'):
                self.log_success(f"Training resumed: {result.get('message', '')}")
            else:
                # Reset training state if resume failed
                self._update_training_state(is_training_active=False)
                self.log_error(f"Training resume failed: {result.get('message', '')}")
                
        except Exception as e:
            # Reset training state on error
            self._update_training_state(is_training_active=False)
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
        """Handle save config button click - delegates to ConfigurationMixin."""
        try:
            self.log_info("Saving configuration...")
            result = self.save_config()  # Delegates to ConfigurationMixin
            
            if result.get('success', True):  # ConfigurationMixin may return True directly
                self.log_success("Configuration saved successfully")
            else:
                self.log_error(f"Save config failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            self.log_error(f"Save config error: {e}")
    
    def _handle_reset_config(self, button=None) -> None:
        """Handle reset config button click - delegates to ConfigurationMixin."""
        try:
            self.log_info("Resetting configuration...")
            result = self.reset_config()  # Delegates to ConfigurationMixin
            
            if result.get('success', True):  # ConfigurationMixin may return True directly
                self.log_success("Configuration reset to defaults")
                # Sync UI after reset
                self._sync_config_to_ui()
            else:
                self.log_error(f"Reset config failed: {result.get('message', 'Unknown error')}")
                
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
                self._update_training_state(has_model=False)
                return {'success': False, 'message': 'No backbone configuration available'}
            
            if not self._config_handler:
                return {'success': False, 'message': 'Config handler not available'}
            
            # Update model selection based on backbone
            model_result = self._config_handler.select_model_from_backbone(backbone_config)
            if model_result['success']:
                # Update configuration using ConfigurationMixin delegation
                for key, value in model_result['model_selection'].items():
                    self.update_config_value(f'model_selection.{key}', value)
                self._sync_config_to_ui()
                
                # Update training state to reflect model availability
                has_model = bool(model_result['model_selection'].get('backbone_type'))
                self._update_training_state(has_model=has_model)
                
                return {
                    'success': True,
                    'message': 'Backbone configuration refreshed successfully',
                    'model_selection': model_result['model_selection']
                }
            else:
                self._update_training_state(has_model=False)
                return model_result
                
        except Exception as e:
            error_msg = f"Failed to refresh backbone config: {e}"
            self.logger.error(error_msg)
            self._update_training_state(has_model=False)
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
    
    # Configuration methods removed - now properly delegated to ConfigurationMixin
    # save_current_config() -> use self.save_config() from ConfigurationMixin
    # reset_to_defaults() -> use self.reset_config() from ConfigurationMixin
    
    def _sync_config_to_ui(self) -> None:
        """Sync current configuration to UI components."""
        try:
            if self._ui_components:
                current_config = self.get_current_config()
                
                # Update form if available
                form_container = self._ui_components.get('form_container')
                if form_container and hasattr(form_container, 'update_from_config'):
                    form_container.update_from_config(current_config)
                    self.log_debug("UI form synchronized with configuration")
                else:
                    self.log_debug("Form container or update method not available")
                
                # Update training state based on configuration changes
                self._update_training_state_from_config(current_config)
                    
        except Exception as e:
            self.log_warning(f"Failed to sync config to UI: {e}")
    
    def _update_training_state_from_config(self, config: Dict[str, Any]) -> None:
        """Update training state based on configuration changes."""
        try:
            # Check for model and checkpoint availability after config sync
            model_selection = config.get('model_selection', {})
            has_model = bool(model_selection.get('backbone_type'))
            has_checkpoint = bool(model_selection.get('checkpoint_path'))
            
            # Update training state based on configuration
            self._update_training_state(has_model=has_model, has_checkpoint=has_checkpoint)
            
        except Exception as e:
            self.log_warning(f"Failed to update training state from config: {e}")

    # _update_module_config method removed - replaced with proper ConfigurationMixin delegation
    # Use self.update_config_value(key, value) or self.update_config_section(section, values) instead
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status including button states."""
        return {
            'initialized': self._config_handler is not None and self._ui_components is not None,
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'config_available': self._config_handler is not None,
            'charts_available': bool(self._chart_updaters),
            'training_state': self._training_state.copy(),
            'button_states': {
                'has_model': self._has_model,
                'has_checkpoint': self._has_checkpoint,
                'is_training_active': self._is_training_active
            }
        }
    
    def _initialize_training_state(self) -> None:
        """
        Initialize training state based on current configuration and available data.
        
        This method checks for model availability, checkpoint existence, and 
        training status to set initial button states correctly.
        """
        try:
            # Check if we have a model from backbone configuration
            backbone_config = self._get_backbone_configuration()
            has_model = bool(backbone_config and backbone_config.get('available_models'))
            
            # Check for existing checkpoints (placeholder for now)
            has_checkpoint = False  # TODO: Implement checkpoint detection
            
            # Training should start as inactive
            is_training_active = False
            
            # Update the training state
            self._update_training_state(
                has_model=has_model,
                has_checkpoint=has_checkpoint,
                is_training_active=is_training_active
            )
            
            self.logger.debug(f"Training state initialized: model={has_model}, checkpoint={has_checkpoint}, active={is_training_active}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize training state: {e}")
            # Set safe defaults on error
            self._update_training_state(
                has_model=False,
                has_checkpoint=False,
                is_training_active=False
            )
    
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
                for _, component in self._ui_components.items():
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
