"""
Model Manager for Training Pipeline

This module handles all model-related operations during training pipeline execution,
following SRP by focusing solely on model building, validation, and configuration.
"""

from typing import Dict, Any, Optional, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.training.phases.mixins.callbacks import CallbacksMixin
from smartcash.model.core.model_utils import ModelUtils
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
import torch

from smartcash.model.core.checkpoints.checkpoint_manager import create_checkpoint_manager

logger = get_logger(__name__)


class ModelManager(CallbacksMixin):
    """
    Manages model building, validation, and configuration for training pipeline.
    
    Responsibilities:
    - Build and validate models using model API
    - Handle model architecture inference and validation
    - Manage memory optimization setup
    - Configure model for different training phases
    
    Does NOT handle:
    - Training loops (delegated to TrainingPhaseManager)
    - Pipeline orchestration (delegated to PipelineOrchestrator)
    - Data loading (delegated to other components)
    """
    
    def __init__(self, progress_tracker, log_callback=None, is_resuming: bool = False):
        """
        Initialize model manager.
        
        Args:
            progress_tracker: Progress tracking instance
            log_callback: Callback for log messages
        """
        super().__init__()
        
        self.progress_tracker = progress_tracker
        self.set_log_callback(log_callback)
        
        # Model state
        self.model_api = None
        self.model = None
        self.memory_optimizer = None
        self.checkpoint_manager = None # Initialize to None

    def setup_model(self, config: Dict[str, Any], use_yolov5_integration: bool = True, is_resuming: bool = False) -> Tuple[Any, Any]:
        """
        Set up model API and build model according to configuration.
        
        Args:
            config: Training configuration
            use_yolov5_integration: Whether to use YOLOv5 integration
            
        Returns:
            Tuple of (model_api, built_model)
        """
        try:
            self.emit_log('info', '🏗️ Setting up model for training pipeline')
            
            # Initialize checkpoint manager here, where config is available
                                    self.checkpoint_manager = create_checkpoint_manager(config, is_resuming=is_resuming)

            # Create model API
            self.model_api = self._create_model_api(config, use_yolov5_integration)

            # Build model
            self.model = self._build_model(config)
            
            # Validate model architecture
            self._validate_model_architecture()
            
            # Setup memory optimization
            self._setup_memory_optimization()
            
            self.emit_log('info', '✅ Model setup completed successfully')
            
            return self.model_api, self.model
            
        except Exception as e:
            self.emit_log('error', f'❌ Model setup failed: {str(e)}')
            raise
    
    def _create_model_api(self, config: Dict[str, Any], use_yolov5_integration: bool) -> Any:
        """Create and initialize model API."""
        self.progress_tracker.start_operation("Model Setup", 4)
        self.progress_tracker.update_operation(1, "🏗️ Creating model API...")
        
        try:
            # Import here to avoid circular dependency
            from smartcash.model.api.core import create_api
            
            model_api = create_api(
                config=config,
                use_yolov5_integration=use_yolov5_integration
            )
            
            self.emit_log('info', '✅ Model API created successfully', {
                'yolov5_integration': use_yolov5_integration,
                'config_keys': list(config.keys())
            })
            
            return model_api
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to create model API: {str(e)}')
            raise
    
    def _build_model(self, config: Dict[str, Any]) -> Any:
        """Build model using model API."""
        self.progress_tracker.update_operation(2, "🔧 Building model architecture...")
        
        try:
            # Get model configuration
            model_config = config.get('model', {})
            
            # Build model
            build_result = self.model_api.build_model(model_config=model_config)
            
            if not build_result['success']:
                raise RuntimeError(f"Model building failed: {build_result.get('error', 'Unknown error')}")
            
            model = build_result['model']
            model_info = build_result.get('model_info', {})
            
            self.progress_tracker.update_operation(3, "✅ Model architecture built")
            
            self.emit_log('info', '✅ Model built successfully', {
                'architecture_type': 'yolov5',
                'backbone': model_info.get('backbone', 'unknown'),
                'parameters': model_info.get('total_parameters', 0),
                'model_size_mb': model_info.get('model_size_mb', 0)
            })
            
            return model
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to build model: {str(e)}')
            raise
    
    def _validate_model_architecture(self):
        """Validate model architecture compatibility."""
        self.progress_tracker.update_operation(3, "🔍 Validating model architecture...")
        
        try:
            # Validate model using model API
            validation_result = self.model_api.validate_model()
            
            if validation_result['success']:
                self.emit_log('info', '✅ Model validation successful', {
                    'architecture_type': 'yolov5',
                    'validation_info': validation_result
                })
            else:
                error_msg = validation_result.get('error', 'Unknown validation error')
                self.emit_log('warning', f'⚠️ Model validation issues: {error_msg}')
                # Don't raise here unless it's critical - continue with training
            
        except Exception as e:
            self.emit_log('warning', f'⚠️ Model validation failed: {str(e)}')
            # Don't raise here - validation failure shouldn't stop training
    
    def _setup_memory_optimization(self):
        """Set up memory optimization for model."""
        self.progress_tracker.update_operation(4, "🧠 Setting up memory optimization...")
        
        try:
            device = next(self.model.parameters()).device
            self.memory_optimizer = get_memory_optimizer(device)
            
            self.progress_tracker.complete_operation(4, "✅ Model setup complete")
            self.progress_tracker.complete_phase({
                'success': True,
                'device': str(device),
                'optimizer_type': type(self.memory_optimizer).__name__
            })
            
            self.emit_log('info', '✅ Memory optimization setup completed', {
                'device': str(device),
                'optimizer_type': type(self.memory_optimizer).__name__
            })
            
        except Exception as e:
            self.emit_log('warning', f'⚠️ Memory optimization setup failed: {str(e)}')
            # Don't raise here - memory optimization is not critical for training
    
    def prepare_model_for_phase(self, phase_num: int, config: Dict[str, Any]) -> Any:
        """
        Prepare model for specific training phase.
        
        Args:
            phase_num: Phase number (1 or 2)
            config: Training configuration
            
        Returns:
            Model prepared for the specified phase
        """
        try:
            self.emit_log('info', f'🔧 Preparing model for Phase {phase_num}')
            
            if phase_num == 2:
                # Phase 2 requires backbone unfreezing
                self.model = ModelUtils.rebuild_model_for_phase2(
                    self.model_api, self.model, config
                )
                self.emit_log('info', '🔥 Model prepared for Phase 2 (backbone unfrozen)')
            else:
                # Phase 1 uses model as-is
                self.emit_log('info', '❄️ Model prepared for Phase 1 (backbone frozen)')
            
            return self.model
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to prepare model for Phase {phase_num}: {str(e)}')
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.model:
            return {}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),
                'architecture': self.model.__class__.__name__,
                'device': str(next(self.model.parameters()).device),
                'memory_optimizer_available': self.memory_optimizer is not None
            }
            
        except Exception as e:
            self.emit_log('warning', f'⚠️ Failed to get model info: {str(e)}')
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up model manager resources."""
        try:
            self.cleanup_callbacks()
            self.model_api = None
            self.model = None
            self.memory_optimizer = None
            
            self.emit_log('info', '🧹 Model manager resources cleaned up')
            
        except Exception as e:
            logger.warning(f'⚠️ Error during model manager cleanup: {e}')