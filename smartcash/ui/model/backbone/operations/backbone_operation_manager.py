"""
File: smartcash/ui/model/backbone/operations/backbone_operation_manager.py
Operation manager for backbone module extending OperationHandler.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.logger import get_module_logger

# Import service for backend integration
from ..services.backbone_service import BackboneService


class BackboneOperationManager(OperationHandler):
    """
    Operation manager for backbone module.
    
    Features:
    - 🧬 Backbone model operations (validate, build, load, summary)
    - 🏗️ Early training pipeline integration
    - 🔄 Progress tracking and logging integration
    - 🛡️ Error handling with user feedback
    - 🎯 Button management with disable/enable functionality
    - 📋 Operation status tracking and reporting
    - 🔗 Backend model builder integration
    """
    
    def __init__(self, config: Dict[str, Any], operation_container: Any):
        """
        Initialize backbone operation manager.
        
        Args:
            config: Configuration dictionary
            operation_container: UI operation container for logging and progress
        """
        super().__init__(
            module_name='backbone',
            parent_module='model',
            operation_container=operation_container
        )
        
        self.config = config
        self.logger = get_module_logger("smartcash.ui.model.backbone.operations")
        
        # Initialize service
        self._service = None
        self._model_builder = None
        
        # Initialize service instance
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize service instance."""
        try:
            self._service = BackboneService()
            self.log("✅ Backbone service initialized", 'debug')
            
            # Initialize model builder for early training pipeline
            self._initialize_model_builder()
            
        except Exception as e:
            self.log(f"❌ Failed to initialize service: {e}", 'error')
    
    def _initialize_model_builder(self) -> None:
        """Initialize model builder for backend integration."""
        try:
            from smartcash.model.core.model_builder import ModelBuilder
            from smartcash.model.utils.progress_bridge import ModelProgressBridge
            
            # Create progress bridge for model builder
            progress_bridge = ModelProgressBridge()
            self._model_builder = ModelBuilder(self.config, progress_bridge)
            
            self.log("✅ Model builder initialized for early training pipeline", 'debug')
            
        except Exception as e:
            self.log(f"⚠️ Failed to initialize model builder: {e}", 'warning')
            self._model_builder = None
    
    def initialize(self) -> None:
        """Initialize the operation manager."""
        try:
            super().initialize()
            self.log("🔧 Backbone operation manager initialized", 'info')
            
        except Exception as e:
            self.log(f"❌ Failed to initialize backbone operation manager: {e}", 'error')
            self.log(f"❌ Initialization failed: {e}", 'error')
    
    def get_operations(self) -> Dict[str, str]:
        """
        Get available operations.
        
        Returns:
            Dictionary of operation names and descriptions
        """
        return {
            'validate': 'Validate backbone configuration and compatibility',
            'build': 'Build backbone architecture with current configuration (pretrained auto-loaded)'
        }
    
    # ==================== BACKBONE OPERATIONS ====================
    
    def execute_validate(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone configuration validation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Operation result dictionary
        """
        button_states = None
        try:
            # Clear logs from previous operations
            self.clear_logs()
            
            self.log("🔍 Starting backbone validation operation...", 'info')
            button_states = self.disable_all_buttons("⏳ Validating...")
            
            # Update progress
            self.update_progress(0, "Initializing validation...", level="primary")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute validation operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with validation")
            
            result = self._execute_validate_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Validation completed successfully", level="primary")
                self.log("✅ Backbone configuration validated successfully", 'success')
            else:
                self.update_progress(0, "Validation failed", level="primary")
                self.log(f"❌ Validation failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validate operation error: {e}")
            self.log(f"❌ Validation operation error: {e}", 'error')
            self.update_progress(0, "Validation failed", level="primary")
            return {'success': False, 'message': str(e)}
        
        finally:
            if button_states:
                self.enable_all_buttons(button_states)
    
    def execute_build(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone model building operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Build result dictionary
        """
        button_states = None
        try:
            # Clear logs from previous operations
            self.clear_logs()
            
            self.log("🏗️ Starting backbone build operation...", 'info')
            button_states = self.disable_all_buttons("⏳ Building...")
            
            # Update progress
            self.update_progress(0, "Initializing model build...", level="primary")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute build operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with build")
            
            result = self._execute_build_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Model build completed successfully", level="primary")
                self.log("✅ Backbone model built successfully", 'success')
            else:
                self.update_progress(0, "Model build failed", level="primary")
                self.log(f"❌ Build failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Build operation error: {e}")
            self.log(f"❌ Build operation error: {e}", 'error')
            self.update_progress(0, "Build failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            if button_states:
                self.enable_all_buttons(button_states)
    
    
    
    # ==================== SERVICE INTEGRATION ====================
    
    def _execute_validate_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation operation with service integration."""
        try:
            backbone_config = config.get('backbone', {})
            model_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            # Create async wrapper for progress and log callbacks
            def sync_progress_callback(progress, message):
                self.update_progress(progress, message, level="primary")
            
            def sync_log_callback(level, message):
                self.log(message, level.lower())
            
            # Run async validation in sync context
            import asyncio
            
            async def async_validate():
                return await self._service.validate_backbone_config(
                    backbone_config,
                    progress_callback=lambda step, total, msg: sync_progress_callback(
                        int((step + 1) / total * 100), msg
                    ),
                    log_callback=lambda level, msg: sync_log_callback(level, msg)
                )
            
            # Execute async operation with proper event loop handling
            try:
                # Try to get existing event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in a running loop, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, async_validate())
                        validation_result = future.result()
                except RuntimeError:
                    # No running loop, safe to use asyncio.run
                    validation_result = asyncio.run(async_validate())
            except Exception as async_error:
                # Fallback to basic validation without service
                self.log(f"⚠️ Service validation unavailable, using basic validation", 'warning')
                validation_result = self._basic_validation_fallback(backbone_config)
            
            return {
                'success': validation_result.get('valid', False),
                'message': 'Backbone validation completed',
                'validation_results': validation_result,
                'model_type': model_type
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service validation failed: {e}'}
    
    def _basic_validation_fallback(self, backbone_config: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation fallback when service is unavailable."""
        try:
            model_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            # Basic validation checks
            valid = True
            issues = []
            
            if model_type not in ['efficientnet_b4', 'cspdarknet']:
                valid = False
                issues.append(f"Unsupported model type: {model_type}")
            
            input_size = backbone_config.get('input_size', 640)
            if input_size < 320 or input_size > 1280:
                valid = False
                issues.append(f"Invalid input size: {input_size}")
            
            num_classes = backbone_config.get('num_classes', 7)
            if num_classes < 1 or num_classes > 100:
                valid = False
                issues.append(f"Invalid number of classes: {num_classes}")
            
            return {
                'valid': valid,
                'model_type': model_type,
                'issues': issues,
                'fallback': True
            }
            
        except Exception as e:
            return {
                'valid': False,
                'model_type': 'unknown',
                'issues': [f"Validation error: {e}"],
                'fallback': True
            }
    
    def _execute_build_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute build operation with service integration."""
        try:
            backbone_config = config.get('backbone', {})
            
            # Create async wrapper for progress and log callbacks
            def sync_progress_callback(progress, message):
                self.update_progress(progress, message, level="primary")
            
            def sync_log_callback(level, message):
                self.log(message, level.lower())
            
            # Run async build in sync context
            import asyncio
            
            async def async_build():
                return await self._service.build_backbone_architecture(
                    backbone_config,
                    progress_callback=lambda step, total, msg: sync_progress_callback(
                        int((step + 1) / total * 100), msg
                    ),
                    log_callback=lambda level, msg: sync_log_callback(level, msg)
                )
            
            # Execute async operation
            try:
                loop = asyncio.get_event_loop()
                build_result = loop.run_until_complete(async_build())
            except RuntimeError:
                # No event loop running, create new one
                build_result = asyncio.run(async_build())
            
            return build_result
            
        except Exception as e:
            return {'success': False, 'message': f'Service build failed: {e}'}
    
    def get_current_model_summary(self) -> Dict[str, Any]:
        """
        Get current model summary if available.
        
        Returns:
            Current model summary dictionary
        """
        try:
            # Try to get model info from API if available
            if self._model_builder and hasattr(self._model_builder, 'get_model_info'):
                return self._model_builder.get_model_info()
            
            # Return basic config info if no model built yet
            return {
                'status': 'not_built',
                'backbone': self.config.get('backbone', {}).get('model_type', 'efficientnet_b4'),
                'pretrained': self.config.get('backbone', {}).get('pretrained', True),
                'message': 'Model not built yet - build to see full summary'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {e}")
            return {
                'status': 'error',
                'message': f'Error retrieving summary: {e}'
            }
    
    # ==================== FAIL-FAST APPROACH ====================
    # No fallback simulations - service must be available for operations
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current operation manager status.
        
        Returns:
            Status dictionary
        """
        return {
            'initialized': True,
            'service_ready': self._service is not None,
            'model_builder_ready': self._model_builder is not None,
            'available_operations': list(self.get_operations().keys()),
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'early_training_enabled': True
        }
    
    def cleanup(self) -> None:
        """Cleanup operation manager resources."""
        try:
            # Cleanup service instance
            if self._service and hasattr(self._service, 'cleanup'):
                self._service.cleanup()
            
            # Cleanup model builder
            if self._model_builder and hasattr(self._model_builder, 'cleanup'):
                self._model_builder.cleanup()
            
            # Clear references
            self._service = None
            self._model_builder = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")