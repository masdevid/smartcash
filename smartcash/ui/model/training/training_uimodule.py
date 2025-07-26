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
from .configs.unified_training_defaults import get_unified_training_defaults, validate_unified_training_config
from .operations.unified_training_operation import UnifiedTrainingOperation

# Import model mixins for shared functionality
from smartcash.ui.model.mixins import (
    ModelConfigSyncMixin,
    BackendServiceMixin
)


class TrainingUIModule(ModelConfigSyncMixin, BackendServiceMixin, BaseUIModule, OperationMixin, ButtonHandlerMixin):
    """
    UIModule implementation for model training using BaseUIModule pattern.
    
    Features:
    - 🚀 Unified training pipeline integration
    - 📊 Simplified configuration form
    - 🔄 Real-time progress tracking
    - 🎯 Direct training pipeline execution
    - 💾 Automatic visualization generation
    - 🛡️ Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize training UI module."""
        super().__init__(
            module_name='training',
            parent_module='model',
            enable_environment=True
        )
        
        self.logger = get_module_logger("smartcash.ui.model.training")
        
        # Lazy initialization flags
        self._ui_components_created = False
        self._config_handler: Optional[TrainingConfigHandler] = None
        self._training_state = {'phase': 'idle'}
        
        # Button registration tracking
        self._buttons_registered = False
        
        # Simplified training state for unified pipeline
        self._has_model = True  # Unified pipeline handles validation
        self._has_checkpoint = False  # Not relevant for unified pipeline
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
        merged_config = config or get_unified_training_defaults()
        config_handler = TrainingConfigHandler(merged_config)
        if hasattr(self, '_ui_components') and self._ui_components:
            config_handler.set_ui_components(self._ui_components)
        return config_handler
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for unified training pipeline."""
        return get_unified_training_defaults()
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create simplified UI components for unified training pipeline with lazy initialization."""
        # Prevent double initialization
        if self._ui_components_created and hasattr(self, '_ui_components') and self._ui_components:
            self.logger.debug("⏭️ Skipping UI component creation - already created")
            return self._ui_components
            
        try:
            from .components.unified_training_ui import create_unified_training_ui
            
            ui_components = create_unified_training_ui(config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Mark as created to prevent reinitalization
            self._ui_components_created = True
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def _get_module_button_handlers(self) -> Dict[str, Callable]:
        """Get module-specific button handlers for unified training."""
        return {
            'start_training': self._handle_start_unified_training,
            'stop_training': self._handle_stop_unified_training,
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
        Setup simplified button dependencies for unified training.
        """
        try:
            # Start training button is always available (unified pipeline handles validation)
            self.set_button_dependency('start_training', self._check_start_training_dependency)
            
            # Stop training button depends on training being active
            self.set_button_dependency('stop_training', self._check_stop_training_dependency)
            
            self.logger.debug("✅ Unified training button dependencies configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup training button dependencies: {e}")
    
    def _check_start_training_dependency(self) -> bool:
        """
        Check if start training button should be enabled.
        
        Returns:
            True if start training should be enabled (not currently training)
        """
        return not self._is_training_active
    
    def _check_stop_training_dependency(self) -> bool:
        """
        Check if stop training button should be enabled.
        
        Returns:
            True if stop training should be enabled (training is active)
        """
        return self._is_training_active
    
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
        Update simplified training button states.
        """
        try:
            # Define button conditions for unified training
            button_conditions = {
                'start_training': not self._is_training_active,
                'stop_training': self._is_training_active
            }
            
            # Define reasons for disabled buttons
            button_reasons = {
                'start_training': "Training in progress" if self._is_training_active else None,
                'stop_training': "No active training to stop" if not self._is_training_active else None
            }
            
            # Update button states using enhanced mixin
            self.update_button_states_based_on_condition(button_conditions, button_reasons)
            
            # Also update UI button states directly
            from .components.unified_training_ui import update_training_buttons_state
            if self._ui_components:
                update_training_buttons_state(
                    self._ui_components, 
                    is_training=self._is_training_active,
                    has_model=True  # Unified pipeline handles model validation
                )
            
            self.logger.debug("Unified training button states updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update training button states: {e}")
    
    def _get_start_training_disable_reason(self) -> Optional[str]:
        """Get reason why start training button is disabled."""
        if self._is_training_active:
            return "Training already in progress"
        return None
    
    def _handle_start_unified_training(self, button=None) -> None:
        """Handle start unified training button click."""
        try:
            # Ignore button parameter - not needed for unified training
            self.log_info("🚀 Starting unified training pipeline...")
            
            # Update state to indicate training is starting
            self._update_training_state(is_training_active=True)
            
            # Get current configuration from form
            current_config = self._get_form_configuration()
            
            # Validate configuration
            validation_result = validate_unified_training_config(current_config)
            if not validation_result['success']:
                error_msg = f"Configuration validation failed: {'; '.join(validation_result['errors'])}"
                self._update_training_state(is_training_active=False)
                self.log_error(error_msg)
                return
            
            # Show warnings if any
            for warning in validation_result.get('warnings', []):
                self.log_warning(f"⚠️ {warning}")
            
            result = self.execute_unified_training(current_config)
            
            if result.get('success'):
                self.log_success(f"🎉 {result.get('message', 'Training completed successfully')}")
                self._handle_training_completion(result)
            else:
                # Reset training state if training failed
                self._update_training_state(is_training_active=False)
                self.log_error(f"❌ Training failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            # Reset training state on error
            self._update_training_state(is_training_active=False)
            self.log_error(f"Unified training error: {e}")
    
    def _handle_stop_unified_training(self, button=None) -> None:
        """Handle stop unified training button click."""
        try:
            # Ignore button parameter - not needed for unified training
            self.log_warning("⚠️ Training stop requested...")
            self.log_info("Note: Unified training pipeline doesn't support mid-training stops")
            self.log_info("Training will complete current epoch and finish normally")
            
            # Note: The unified training pipeline doesn't currently support stopping
            # This is a limitation that would require keyboard interrupt simulation
            
        except Exception as e:
            self.log_error(f"Stop training error: {e}")
    
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
    
    # ==================== UNIFIED TRAINING METHODS ====================
    
    def execute_unified_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unified training pipeline."""
        try:
            # Create unified training operation
            operation = UnifiedTrainingOperation(
                ui_module=self,
                config=config,
                callbacks=self._get_operation_callbacks()
            )
            
            # Execute the training
            return operation.execute()
            
        except Exception as e:
            error_msg = f"Unified training execution failed: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def _get_form_configuration(self) -> Dict[str, Any]:
        """Get current configuration from form."""
        try:
            if self._ui_components:
                get_form_values = self._ui_components.get('get_form_values')
                if get_form_values:
                    form_values = get_form_values()
                    # Wrap form values in training section
                    return {'training': form_values}
            
            # Fallback to current config
            return self.get_current_config()
            
        except Exception as e:
            self.logger.warning(f"Failed to get form configuration: {e}")
            return self.get_current_config()
    
    def _handle_training_completion(self, result: Dict[str, Any]) -> None:
        """Handle training completion and update UI."""
        try:
            # Update training state
            self._update_training_state(is_training_active=False)
            
            # Update button states
            from .components.unified_training_ui import update_training_buttons_state, update_summary_display
            
            if self._ui_components:
                update_training_buttons_state(self._ui_components, is_training=False, has_model=True)
                update_summary_display(self._ui_components, result)
            
        except Exception as e:
            self.logger.warning(f"Failed to handle training completion: {e}")
    
    def _get_operation_callbacks(self) -> Dict[str, Callable]:
        """Get callbacks for unified training operation."""
        return {
            'on_progress': self._handle_operation_progress,
            'on_success': self._handle_operation_success,
            'on_failure': self._handle_operation_failure
        }
    
    def _handle_operation_progress(self, progress: int, message: str = "") -> None:
        """Handle operation progress updates."""
        try:
            if self._ui_components and 'main_container' in self._ui_components:
                main_container = self._ui_components['main_container']
                if hasattr(main_container, 'update_progress'):
                    main_container.update_progress(progress, message)
        except Exception as e:
            self.log_warning(f"Failed to update progress: {e}")
    
    def _handle_operation_success(self, message: str) -> None:
        """Handle operation success."""
        self.log_success(message)
    
    def _handle_operation_failure(self, message: str) -> None:
        """Handle operation failure."""
        self.log_error(message)
    
    def _sync_config_to_ui(self) -> None:
        """Sync current configuration to unified training form."""
        try:
            if self._ui_components:
                current_config = self.get_current_config()
                
                # Update unified form if available
                update_form_from_config = self._ui_components.get('update_form_from_config')
                if update_form_from_config:
                    update_form_from_config(current_config)
                    self.log_debug("Unified form synchronized with configuration")
                    
        except Exception as e:
            self.log_warning(f"Failed to sync config to UI: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current unified training status."""
        return {
            'initialized': self._config_handler is not None and self._ui_components is not None,
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'config_available': self._config_handler is not None,
            'training_state': self._training_state.copy(),
            'button_states': {
                'is_training_active': self._is_training_active
            },
            'pipeline_type': 'unified'
        }
    
    def _initialize_training_state(self) -> None:
        """
        Initialize simplified training state for unified pipeline.
        """
        try:
            # Unified pipeline doesn't require model/checkpoint checks upfront
            # It handles validation internally during execution
            self._update_training_state(
                has_model=True,  # Assume available - pipeline will validate
                has_checkpoint=False,  # Not relevant for unified pipeline
                is_training_active=False
            )
            
            self.logger.debug("Unified training state initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize training state: {e}")
            # Set safe defaults on error
            self._update_training_state(
                has_model=True,
                has_checkpoint=False,
                is_training_active=False
            )
    
    def cleanup(self) -> None:
        """Simplified cleanup for unified training UI."""
        try:
            # Clear training state
            self._training_state.clear()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
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
            
            if hasattr(self, 'logger'):
                self.logger.info("Unified training module cleanup completed")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Unified training module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion
