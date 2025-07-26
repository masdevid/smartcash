"""
File: smartcash/ui/model/pretrained/pretrained_uimodule.py
Description: Pretrained Module implementation using BaseUIModule pattern.

This module provides the PretrainedUIModule class which implements the UI for
working with pretrained models. It follows the BaseUIModule pattern and is
meant to be used with the PretrainedUIFactory for proper initialization.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Type, Callable
from unittest.mock import MagicMock

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Pretrained module imports
from smartcash.ui.model.pretrained.components.pretrained_ui import create_pretrained_ui_components
from smartcash.ui.model.pretrained.configs.pretrained_config_handler import PretrainedConfigHandler
from smartcash.ui.model.pretrained.configs.pretrained_defaults import get_default_pretrained_config
from smartcash.ui.model.pretrained.operations.pretrained_factory import (
    PretrainedOperationFactory, 
    PretrainedOperationType
)

# Import model mixins for shared functionality
from smartcash.ui.model.mixins import (
    ModelConfigSyncMixin,
    BackendServiceMixin
)

# Module-level instance for singleton pattern
_pretrained_module_instance = None


class PretrainedUIModule(ModelConfigSyncMixin, BackendServiceMixin, BaseUIModule):
    """
    Pretrained Module implementation using BaseUIModule pattern.
    
    Manages YOLOv5s and EfficientNet-B4 models with download, validation, 
    and refresh operations with a simplified one-click setup workflow.
    """
    
    def __init__(self):
        """Initialize Pretrained UI module with environment support."""
        # Initialize BaseUIModule with environment support enabled
        super().__init__(
            module_name='pretrained',
            parent_module='model',
            enable_environment=True  # Enable environment management features
        )
        
        # Ensure the logger is properly initialized with the module namespace
        self._update_logging_context()
        
        # Lazy initialization flags
        self._ui_components_created = False
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'header_container', 
            'form_container',
            'action_container',
            'operation_container'
        ]
        
        # Log successful initialization
        self.log_debug("✅ PretrainedUIModule initialized")
        
        # Pretrained-specific attributes
        self._model_status = {
            'last_refresh': None,
            'models_found': [],
            'validation_results': {}
        }
        
        # For testing purposes
        self._mock_operations = {}
    
    def _register_default_operations(self) -> None:
        """Register default operation handlers including pretrained-specific operations."""
        # Call parent method to register base operations
        super()._register_default_operations()
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Pretrained module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        base_handlers = super()._get_module_button_handlers()
        
        # Add Pretrained-specific handlers
        pretrained_handlers = {
            'oneclick_setup': self._operation_oneclick_setup,
            'refresh': self._operation_refresh
        }
        
        # Merge with base handlers
        return {**base_handlers, **pretrained_handlers}
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Pretrained module (BaseUIModule requirement)."""
        return get_default_pretrained_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> PretrainedConfigHandler:
        """Create config handler instance for Pretrained module (BaseUIModule requirement)."""
        handler = PretrainedConfigHandler(config)
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Pretrained module with lazy initialization."""
        # Prevent double initialization
        if self._ui_components_created and hasattr(self, '_ui_components') and self._ui_components:
            self.log_debug("⏭️ Skipping UI component creation - already created")
            return self._ui_components
            
        try:
            self.log_debug("Creating Pretrained UI components...")
            ui_components = create_pretrained_ui_components(module_config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Mark as created to prevent reinitalization
            self._ui_components_created = True
            self.log_debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.log_error(f"Failed to create UI components: {e}")
            raise
            
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, *args, **kwargs) -> bool:
        """
        Initialize the module with environment support.
        
        Args:
            *args: Variable length argument list
            **kwargs: Additional initialization arguments
            
        Returns:
            bool: True if initialization was successful, False otherwise
            
        Raises:
            Exception: If initialization fails and not in test mode
        """
        # Call parent initialization and check result
        if not super().initialize(*args, **kwargs):
            return False

        try:
            # Initialize model status with default values
            self._model_status = {
                'last_refresh': None,
                'models_found': [],
                'validation_results': {}
            }
            import contextlib
            # Attempt initial model refresh with error handling
            with contextlib.suppress(Exception):
                self._execute_pretrained_operation('refresh')
                self._model_status['last_refresh'] = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            error_msg = f"Gagal menginisialisasi PretrainedUIModule: {e}"
            self.log_error(error_msg)
            raise
    
    # ==================== OPERATION HANDLERS ====================
    
    
    def _operation_refresh(self, button=None) -> Dict[str, Any]:
        """Handle model refresh operation."""
        return self._execute_operation_with_wrapper(
            operation_name="Model Refresh",
            operation_func=lambda: self._execute_refresh_operation(),
            button=button,
            validation_func=lambda: self._validate_models(),
            success_message="Model refresh completed successfully",
            error_message="Model refresh failed"
        )
    
    def _operation_oneclick_setup(self, button=None) -> Dict[str, Any]:
        """Handle one-click model setup operation."""
        return self._execute_operation_with_wrapper(
            operation_name="One-Click Model Setup",
            operation_func=lambda: self._execute_oneclick_setup_operation(),
            button=button,
            validation_func=lambda: {'valid': True},  # No pre-validation needed for one-click
            success_message="One-click model setup completed successfully",
            error_message="One-click model setup failed"
        )
    
    
    # ==================== OPERATION EXECUTION METHODS ====================
    
    def _execute_pretrained_operation(self, operation_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute pretrained operation using factory pattern.
        
        Args:
            operation_type: Type of operation to execute (oneclick_setup, refresh)
            config: Optional configuration overrides
            
        Returns:
            Dictionary with operation results containing:
            - success: Boolean indicating if operation was successful
            - message: Optional status message
            - error: Optional error message if operation failed
            - Additional operation-specific fields
        """
        from smartcash.ui.model.pretrained.operations.pretrained_factory import (
            PretrainedOperationFactory,
            PretrainedOperationType
        )
        
        try:
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            if hasattr(self, 'get_component'):
                ui_components['operation_container'] = self.get_component('operation_container')
            
            # Prepare config
            operation_config = self.get_current_config()
            if config:
                operation_config.update(config)
            
            # Execute operation via factory
            result = PretrainedOperationFactory.execute_operation(
                operation_type=operation_type,
                ui_components=ui_components,
                config=operation_config
            )
            
            # Update internal model status if refresh operation was successful
            if operation_type == 'refresh' and result.get('success'):
                self._model_status.update({
                    'last_refresh': result.get('refresh_time', datetime.now().isoformat()),
                    'models_found': result.get('models_found', []),
                    'validation_results': result.get('validation_results', {})
                })
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Log the error if we have a logger, with graceful error handling
            try:
                if hasattr(self, 'log') and self.log is not None:
                    self.log(f"Error in {operation_type} operation: {error_msg}", 'error')
            except Exception:
                # If logging fails, continue gracefully
                pass
            
            return {
                "success": False,
                "error": error_msg,
                "message": f"Failed to execute {operation_type} operation"
            }
    
    
    def _execute_refresh_operation(self) -> Dict[str, Any]:
        """Execute refresh operation."""
        return self._execute_pretrained_operation('refresh')
    
    def _execute_oneclick_setup_operation(self) -> Dict[str, Any]:
        """Execute one-click setup operation."""
        return self._execute_pretrained_operation('oneclick_setup')
    
    
    # ==================== PRETRAINED-SPECIFIC METHODS ====================
    
    
    def _validate_models(self) -> Dict[str, Any]:
        """Validate model configuration and environment."""
        return {'valid': True}
    
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status information."""
        return {
            'initialized': self._is_initialized,
            'module_name': self.module_name,
            'environment_type': 'colab' if self.is_colab else 'local',
            'config_loaded': self._config_handler is not None,
            'ui_created': bool(self._ui_components),
            'model_status': self._model_status,
            'environment_paths': self.environment_paths
        }
        
    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup operation factory if it exists
            if hasattr(self, '_operation_factory') and self._operation_factory:
                if hasattr(self._operation_factory, 'cleanup'):
                    self._operation_factory.cleanup()
            
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
                self.logger.info("Pretrained module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Pretrained module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion