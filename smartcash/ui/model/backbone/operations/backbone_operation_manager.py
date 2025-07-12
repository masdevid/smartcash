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
            'build': 'Build backbone architecture with current configuration',
            'load': 'Load pretrained backbone model from existing checkpoint',
            'summary': 'Generate detailed model summary and statistics'
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
        try:
            self.log("🔍 Starting backbone validation operation...", 'info')
            self.disable_buttons(['validate', 'build', 'load', 'summary'])
            
            # Update progress
            self.update_progress(0, "Initializing validation...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute validation operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with validation")
            
            result = self._execute_validate_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Validation completed successfully")
                self.log("✅ Backbone configuration validated successfully", 'success')
            else:
                self.update_progress(0, "Validation failed")
                self.log(f"❌ Validation failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validate operation error: {e}")
            self.log(f"❌ Validation operation error: {e}", 'error')
            self.update_progress(0, "Validation failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['validate', 'build', 'load', 'summary'])
    
    def execute_build(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute backbone model building operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Build result dictionary
        """
        try:
            self.log("🏗️ Starting backbone build operation...", 'info')
            self.disable_buttons(['validate', 'build', 'load', 'summary'])
            
            # Update progress
            self.update_progress(0, "Initializing model build...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute build operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with build")
            
            result = self._execute_build_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Model build completed successfully")
                self.log("✅ Backbone model built successfully", 'success')
            else:
                self.update_progress(0, "Model build failed")
                self.log(f"❌ Build failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Build operation error: {e}")
            self.log(f"❌ Build operation error: {e}", 'error')
            self.update_progress(0, "Build failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['validate', 'build', 'load', 'summary'])
    
    def execute_load(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute pretrained backbone loading operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Load result dictionary
        """
        try:
            self.log("📥 Starting backbone load operation...", 'info')
            self.disable_buttons(['validate', 'build', 'load', 'summary'])
            
            # Update progress
            self.update_progress(0, "Initializing model loading...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute load operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with load")
            
            result = self._execute_load_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Model loaded successfully")
                self.log("✅ Pretrained backbone loaded successfully", 'success')
            else:
                self.update_progress(0, "Model loading failed")
                self.log(f"❌ Load failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Load operation error: {e}")
            self.log(f"❌ Load operation error: {e}", 'error')
            self.update_progress(0, "Load failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['validate', 'build', 'load', 'summary'])
    
    def execute_summary(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute model summary generation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Summary result dictionary
        """
        try:
            self.log("📊 Starting model summary generation...", 'info')
            self.disable_buttons(['summary'])
            
            # Update progress
            self.update_progress(0, "Generating model summary...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute summary operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with summary")
            
            result = self._execute_summary_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Summary generated successfully")
                self.log("✅ Model summary generated successfully", 'success')
            else:
                self.update_progress(0, "Summary generation failed")
                self.log(f"❌ Summary failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Summary operation error: {e}")
            self.log(f"❌ Summary operation error: {e}", 'error')
            self.update_progress(0, "Summary failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['summary'])
    
    # ==================== SERVICE INTEGRATION ====================
    
    def _execute_validate_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation operation with service integration."""
        try:
            backbone_config = config.get('backbone', {})
            model_type = backbone_config.get('model_type', 'efficientnet_b4')
            
            # Validate configuration
            self.update_progress(20, "Validating configuration...")
            validation_result = self._service.validate_config(
                backbone_config,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            # Check backbone compatibility
            self.update_progress(60, "Checking backbone compatibility...")
            compatibility_result = self._service.check_backbone_compatibility(
                model_type,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            return {
                'success': validation_result.get('valid', False) and compatibility_result.get('compatible', False),
                'message': 'Backbone validation completed',
                'validation_results': validation_result,
                'compatibility_results': compatibility_result,
                'model_type': model_type
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service validation failed: {e}'}
    
    def _execute_build_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute build operation with service integration."""
        try:
            backbone_config = config.get('backbone', {})
            model_config = config.get('model', {})
            
            # Use model builder for early training pipeline
            if self._model_builder:
                self.update_progress(20, "Building backbone with model builder...")
                
                backbone_type = backbone_config.get('model_type', 'efficientnet_b4')
                detection_layers = backbone_config.get('detection_layers', ['banknote'])
                layer_mode = backbone_config.get('layer_mode', 'single')
                num_classes = backbone_config.get('num_classes', 7)
                input_size = backbone_config.get('input_size', 640)
                feature_optimization = backbone_config.get('feature_optimization', {})
                
                # Build model using backend model builder
                self.update_progress(50, "Constructing model architecture...")
                model = self._model_builder.build(
                    backbone=backbone_type,
                    detection_layers=detection_layers,
                    layer_mode=layer_mode,
                    num_classes=num_classes,
                    img_size=input_size,
                    feature_optimization=feature_optimization
                )
                
                self.update_progress(80, "Calculating model parameters...")
                
                # Get model statistics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                return {
                    'success': True,
                    'message': 'Backbone model built successfully',
                    'model': model,
                    'model_stats': {
                        'total_parameters': total_params,
                        'trainable_parameters': trainable_params,
                        'backbone_type': backbone_type,
                        'input_size': input_size,
                        'num_classes': num_classes
                    },
                    'backbone_config': backbone_config
                }
            else:
                # Fallback to service build
                return self._service.build_backbone(
                    backbone_config,
                    progress_callback=self.update_progress,
                    log_callback=self.log
                )
            
        except Exception as e:
            return {'success': False, 'message': f'Service build failed: {e}'}
    
    def _execute_load_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute load operation with service integration."""
        try:
            backbone_config = config.get('backbone', {})
            model_type = backbone_config.get('model_type', 'efficientnet_b4')
            pretrained = backbone_config.get('pretrained', True)
            
            # Load pretrained backbone
            self.update_progress(30, "Loading pretrained model...")
            load_result = self._service.load_pretrained_backbone(
                model_type,
                pretrained=pretrained,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            # Validate from existing pretrained model if available
            if load_result.get('success') and backbone_config.get('early_training', {}).get('validation_from_pretrained'):
                self.update_progress(70, "Validating from pretrained model...")
                validation_result = self._service.validate_from_pretrained(
                    load_result.get('model'),
                    backbone_config,
                    progress_callback=self.update_progress,
                    log_callback=self.log
                )
                load_result['validation_from_pretrained'] = validation_result
            
            return load_result
            
        except Exception as e:
            return {'success': False, 'message': f'Service load failed: {e}'}
    
    def _execute_summary_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summary operation with service integration."""
        try:
            backbone_config = config.get('backbone', {})
            
            # Generate model summary
            self.update_progress(30, "Analyzing model architecture...")
            summary_result = self._service.generate_model_summary(
                backbone_config,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            return {
                'success': True,
                'message': 'Model summary generated',
                'summary': summary_result,
                'backbone_config': backbone_config
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service summary failed: {e}'}
    
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