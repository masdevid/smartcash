"""
File: smartcash/ui/model/pretrained/pretrained_uimodule.py
Description: Pretrained Module implementation using BaseUIModule pattern.
"""

from typing import Dict, Any, Optional

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Pretrained module imports
from smartcash.ui.model.pretrained.components.pretrained_ui import create_pretrained_ui_components
from smartcash.ui.model.pretrained.configs.pretrained_config_handler import PretrainedConfigHandler
from smartcash.ui.model.pretrained.configs.pretrained_defaults import get_default_pretrained_config


class PretrainedUIModule(BaseUIModule):
    """
    Pretrained Module implementation using BaseUIModule pattern.
    
    Manages YOLOv5s and EfficientNet-B4 models with download, validation, 
    refresh, and cleanup operations.
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
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'header_container', 
            'form_container',
            'action_container',
            'operation_container'
        ]
        
        # Log successful initialization
        self.logger.debug("✅ PretrainedUIModule initialized")
        
        # Pretrained-specific attributes
        self._model_status = {}
    
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
            'download': self._operation_download,
            'validate': self._operation_validate,
            'refresh': self._operation_refresh,
            'cleanup': self._operation_cleanup
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
        """Create UI components for Pretrained module (BaseUIModule requirement)."""
        try:
            self.logger.debug("Creating Pretrained UI components...")
            ui_components = create_pretrained_ui_components(module_config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
            
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Pretrained module with environment detection.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = BaseUIModule.initialize(self)
            
            if success:
                # Post-initialization logging
                self._log_initialization_complete()
                
                # Set global instance for convenience access
                global _pretrained_module_instance
                _pretrained_module_instance = self
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pretrained module: {e}")
            return False
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container."""
        try:
            # Log environment info if environment support is enabled
            if self.has_environment_support:
                env_type = "Google Colab" if self.is_colab else "Local/Jupyter"
                self.log(f"🌍 Environment detected: {env_type}", 'info')
                
                # Safely access environment_paths attributes
                if hasattr(self, 'environment_paths') and self.environment_paths is not None:
                    if hasattr(self.environment_paths, 'data_root') and self.environment_paths.data_root:
                        self.log(f"📁 Working directory: {self.environment_paths.data_root}", 'info')
            
            # Update status panel
            self.log("📊 Status: Ready for pretrained model management", 'info')
            
        except Exception as e:
            self.logger.error(f"Failed to log initialization complete: {e}", exc_info=True)
    
    # ==================== OPERATION HANDLERS ====================
    
    def _operation_download(self, button=None) -> Dict[str, Any]:
        """Handle model download operation."""
        self._execute_operation(
            operation_name="download",
            validate_callback=lambda: self._validate_models(),
            execute_callback=lambda: self._execute_download_operation({})
        )
    
    def _operation_validate(self, button=None) -> Dict[str, Any]:
        """Handle model validation operation."""
        self._execute_operation(
            operation_name="validate",
            validate_callback=lambda: self._validate_models(),
            execute_callback=lambda: self._execute_validate_operation()
        )
    
    def _operation_refresh(self, button=None) -> Dict[str, Any]:
        """Handle model refresh operation."""
        self._execute_operation(
            operation_name="refresh",
            validate_callback=lambda: self._validate_models(),
            execute_callback=lambda: self._execute_refresh_operation()
        )
    
    def _operation_cleanup(self, button=None) -> Dict[str, Any]:
        """Handle model cleanup operation."""
        self._execute_operation(
            operation_name="cleanup",
            validate_callback=lambda: {'valid': True},
            execute_callback=lambda: self._execute_cleanup_operation()
        )
    
    # ==================== OPERATION EXECUTION METHODS ====================
    
    def _execute_pretrained_operation(self, operation_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute pretrained operation using factory pattern."""
        try:
            from smartcash.ui.model.pretrained.operations import PretrainedOperationFactory
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Prepare config
            operation_config = self.get_current_config()
            if config:
                operation_config.update(config)
            
            # Execute operation via factory
            result = PretrainedOperationFactory.execute_operation(
                operation_type, 
                ui_components, 
                operation_config
            )
            
            # Update internal model status if refresh operation was successful
            if operation_type == 'refresh' and result.get('success'):
                self._model_status = {
                    'last_refresh': result.get('refresh_time'),
                    'models_found': result.get('models_found', []),
                    'validation_results': result.get('validation_results', {})
                }
            
            return result
            
        except Exception as e:
            self.log(f"Error in {operation_type} operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_download_operation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download operation."""
        return self._execute_pretrained_operation('download', config)
    
    def _execute_validate_operation(self) -> Dict[str, Any]:
        """Execute validation operation."""
        return self._execute_pretrained_operation('validate')
    
    def _execute_refresh_operation(self) -> Dict[str, Any]:
        """Execute refresh operation."""
        return self._execute_pretrained_operation('refresh')
    
    def _execute_cleanup_operation(self) -> Dict[str, Any]:
        """Execute cleanup operation."""
        return self._execute_pretrained_operation('cleanup')
    
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


# ==================== FACTORY FUNCTIONS ====================

from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

# Create the initialize function using enhanced factory pattern
initialize_pretrained_ui = EnhancedUIModuleFactory.create_display_function(PretrainedUIModule)

# Global module instance for singleton pattern
_pretrained_module_instance: Optional[PretrainedUIModule] = None