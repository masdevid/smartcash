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
from smartcash.model.architectures.model import SmartCashYOLOv5Model
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

    def setup_model(self, config: Dict[str, Any], is_resuming: bool = False) -> Tuple[Any, Any]:
        """
        Set up and build the SmartCashYOLOv5Model.
        
        Args:
            config: Training configuration
            is_resuming: Whether this is a resume operation
            
        Returns:
            Tuple of (model_api, built_model)
        """
        try:
            self.emit_log('info', '🚀 Setting up SmartCashYOLOv5Model...')
            
            # Create and configure the model
            self.model = self._create_smartcash_model_directly(config)
            self.model_api = self._create_smartcash_api_wrapper(config)
            
            # Build and validate model
            build_result = self._build_model(config)
            if not build_result.get('success', False):
                raise RuntimeError(f"Model building failed: {build_result.get('error', 'Unknown error')}")
                
            self._validate_model_architecture()
            
            # Set up memory optimization
            self._setup_memory_optimization()
            
            # Set up checkpoint manager
            self._setup_checkpoint_manager(config)
            
            # Ensure the model is properly initialized in the API wrapper
            if not hasattr(self.model_api, 'model'):
                self.model_api.model = self.model
                
            # Set the log callback on the API wrapper
            self.model_api.log_callback = self.log_callback
                
            self.emit_log('info', '✅ SmartCashYOLOv5Model setup completed successfully')
            
            # Only return the API wrapper, which has all necessary methods
            return self.model_api, self.model_api.model
            
        except Exception as e:
            self.emit_log('error', f'❌ Model setup failed: {str(e)}')
            raise
    
    def _create_smartcash_model_directly(self, config: Dict[str, Any]) -> SmartCashYOLOv5Model:
        """Create and initialize the SmartCashYOLOv5Model."""
        try:
            model_config = config.get('model', {})
            
            # Extract model parameters
            backbone = model_config.get('backbone', 'yolov5s')
            num_classes = 17  # Fixed for SmartCash architecture
            img_size = model_config.get('img_size', 640)
            pretrained = model_config.get('pretrained', True)
            device = config.get('device', {}).get('type', 'auto')
            
            # Create the model
            model = SmartCashYOLOv5Model(
                backbone=backbone,
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                device=device
            )
            
            self.emit_log('info', '✅ SmartCashYOLOv5Model created', {
                'backbone': backbone,
                'img_size': img_size,
                'pretrained': pretrained,
                'device': device
            })
            
            return model
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to create model: {str(e)}')
            raise
    
    def _create_smartcash_api_wrapper(self, config: Dict[str, Any]) -> Any:
        """Create a lightweight API wrapper for SmartCashYOLOv5Model."""
        
        class SmartCashAPIWrapper:
            """Lightweight API wrapper for SmartCashYOLOv5Model."""
            
            def __init__(self, model: SmartCashYOLOv5Model, config: Dict[str, Any]):
                self.model = model
                self.config = config
                self.is_model_built = True
                self.log_callback = None  # Will be set by ModelManager
                
            def __call__(self, *args, **kwargs):
                """Forward call to the underlying model."""
                return self.model(*args, **kwargs)
                
            def get_backbone_name(self) -> str:
                """Get the backbone architecture name."""
                return getattr(self.model, 'backbone_type', 'yolov5')
                
            def get(self, key, default=None):
                """Dictionary-like get method for compatibility."""
                return getattr(self, key, default)
                
            def validate_model(self) -> Dict[str, Any]:
                """Validate the model with a test forward pass."""
                try:
                    device = getattr(self.model, 'device', 'cpu')
                    if isinstance(device, torch.device):
                        device = str(device)
                    
                    dummy_input = torch.randn(1, 3, 640, 640).to(device)
                    with torch.no_grad():
                        output = self.model(dummy_input)
                        
                    # Handle different output formats
                    if isinstance(output, (list, tuple)):
                        output_shape = [o.shape if hasattr(o, 'shape') else type(o).__name__ for o in output]
                    elif hasattr(output, 'shape'):
                        output_shape = output.shape
                    else:
                        output_shape = str(type(output))
                    
                    return {
                        'valid': True,
                        'input_shape': (1, 3, 640, 640),
                        'output_shape': output_shape,
                        'device': device
                    }
                except Exception as e:
                    return {
                        'valid': False,
                        'error': str(e),
                        'exception_type': type(e).__name__
                    }
                    
            def emit_log(self, level: str, message: str, **kwargs):
                """Emit a log message through the parent's log callback."""
                if self.log_callback:
                    self.log_callback(level, message, **kwargs)
                else:
                    logger.log(level.upper(), message, **kwargs)
                    
            def to(self, device):
                """Move model to specified device."""
                try:
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(device)
                        return self
                    else:
                        self.emit_log('warning', 'Model does not support .to() method')
                        return self
                except Exception as e:
                    self.emit_log('error', f'Failed to move model to device {device}: {str(e)}')
                    return self
                    
            def cpu(self):
                """Move model to CPU."""
                return self.to('cpu')
                
            def cuda(self):
                """Move model to CUDA."""
                return self.to('cuda')
                
            def eval(self):
                """Set model to evaluation mode."""
                try:
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                        return self
                    else:
                        self.emit_log('warning', 'Model does not support .eval() method')
                        return self
                except Exception as e:
                    self.emit_log('error', f'Failed to set model to eval mode: {str(e)}')
                    return self
                    
            def train(self, mode=True):
                """Set model to training mode."""
                try:
                    if hasattr(self.model, 'train'):
                        self.model.train(mode)
                        return self
                    else:
                        self.emit_log('warning', 'Model does not support .train() method')
                        return self
                except Exception as e:
                    self.emit_log('error', f'Failed to set model to train mode: {str(e)}')
                    return self
                
            def cleanup(self):
                """Clean up model resources."""
                try:
                    if hasattr(self.model, 'cpu'):
                        self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return True
                except Exception as e:
                    self.emit_log('warning', f'Error during model cleanup: {str(e)}')
                    return False
                    
            def build_model(self, model_config=None):
                """Build the model with the given configuration."""
                try:
                    # If model is already built, just return success
                    if self.is_model_built:
                        # Safe parameter calculation
                        try:
                            if hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                                # YOLO model
                                total_params = sum(p.numel() for p in self.model.model.parameters())
                                trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                                size_mb = sum(p.numel() * p.element_size() for p in self.model.model.parameters()) / (1024 * 1024)
                            else:
                                # Custom backbone
                                total_params = sum(p.numel() for p in self.model.parameters())
                                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                                size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
                        except Exception as param_error:
                            # Fallback if parameter calculation fails
                            total_params = 0
                            trainable_params = 0
                            size_mb = 0.0
                            logger.warning(f"Parameter calculation failed: {param_error}")
                        
                        return {
                            'success': True,
                            'model': self.model,
                            'model_info': {
                                'backbone': getattr(self.model, 'backbone_type', 'yolov5'),
                                'total_parameters': trainable_params,
                                'model_size_mb': size_mb
                            }
                        }
                        
                    # If we need to build the model, implement that logic here
                    # For now, just mark as built and return success
                    self.is_model_built = True
                    return self.build_model(model_config)  # Recursively call after setting built flag
                    
                except Exception as e:
                    import traceback
                    error_msg = f'Failed to build model: {str(e)}'
                    error_traceback = traceback.format_exc()
                    logger.error(f"Build model error: {error_msg}")
                    logger.error(f"Full traceback: {error_traceback}")
                    
                    # Also emit log for the callback system
                    if hasattr(self, 'log_callback') and self.log_callback:
                        self.log_callback('error', error_msg)
                        
                    return {
                        'success': False,
                        'error': error_msg,
                        'exception_type': type(e).__name__,
                        'traceback': error_traceback
                    }
        
        return SmartCashAPIWrapper(self.model, config)
    
    def _build_model(self, config: Dict[str, Any]) -> Any:
        """Build model using model API."""
        self.progress_tracker.update_operation(2, "🔧 Building model architecture...")
        
        try:
            # Get model configuration
            model_config = config.get('model', {})
            
            # Build model using the API wrapper
            build_result = self.model_api.build_model(model_config=model_config)
            
            if not build_result['success']:
                raise RuntimeError(f"Model building failed: {build_result.get('error', 'Unknown error')}")
            
            # Ensure the model is set in the API wrapper
            if 'model' in build_result:
                self.model = build_result['model']
                
                # Ensure the model is set in the API wrapper
                if hasattr(self.model_api, 'model'):
                    self.model_api.model = self.model
            
            model_info = build_result.get('model_info', {})
            
            self.progress_tracker.update_operation(3, "✅ Model architecture built")
            
            self.emit_log('info', '✅ Model built successfully', {
                'architecture_type': 'yolov5',
                'backbone': model_info.get('backbone', 'unknown'),
                'parameters': model_info.get('total_parameters', 0),
                'model_size_mb': model_info.get('model_size_mb', 0)
            })
            
            # Return success result with API wrapper
            return {
                'success': True,
                'model_api': self.model_api,
                'model_info': model_info
            }
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to build model: {str(e)}')
            raise
    
    def _setup_checkpoint_manager(self, config: Dict[str, Any]):
        """Setup checkpoint manager for the training session."""
        try:
            # Create checkpoint manager 
            self.checkpoint_manager = create_checkpoint_manager(
                config=config,
                is_resuming=False
            )
            
            self.emit_log('info', '✅ Checkpoint manager initialized')
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to setup checkpoint manager: {str(e)}')
            # Don't raise - checkpoint manager is not critical for training to start
            self.checkpoint_manager = None
    
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
    
    def prepare_model_for_phase(self, phase_num: int, config: Dict[str, Any], is_resuming: bool = False) -> Any:
        """
        Prepare model for specific training phase with backbone consistency checks.
        
        Args:
            phase_num: Phase number (1 or 2)
            config: Training configuration
            is_resuming: Whether this is a resume operation
            
        Returns:
            Model prepared for the specified phase
        """
        try:
            self.emit_log('info', f'🔧 Preparing model for Phase {phase_num}')
            
            # Check if we're using SmartCash architecture
            is_smartcash_model = isinstance(self.model, SmartCashYOLOv5Model)
            
            if phase_num == 2:
                # Phase 2 special handling
                if is_resuming:
                    # When resuming Phase 2, first load Phase 1's best checkpoint
                    self.emit_log('info', '🔄 Loading Phase 1 best checkpoint for Phase 2 resume')
                    
                    # Get Phase 1's best checkpoint
                    phase1_checkpoint = self.checkpoint_manager.best_metrics_manager._find_best_checkpoint_for_phase(1)
                    if phase1_checkpoint:
                        # Load Phase 1's best model state
                        self.model.load_state_dict(torch.load(phase1_checkpoint)['model_state_dict'])
                        self.emit_log('info', f'✅ Loaded Phase 1 best checkpoint: {phase1_checkpoint.name}')
                    else:
                        raise RuntimeError('❌ No Phase 1 best checkpoint found for Phase 2 resume')
                
                # Verify backbone architecture consistency
                current_backbone = self.model_api.get_backbone_name()
                phase1_backbone = config.get('model', {}).get('backbone', 'unknown')
                
                if current_backbone != phase1_backbone:
                    raise ValueError(
                        f'❌ Backbone mismatch between phases: '
                        f'Phase 1 used {phase1_backbone}, but current is {current_backbone}'
                    )
                
                if is_smartcash_model:
                    # For SmartCash model, use its built-in phase 2 setup
                    self.model.setup_phase_2()
                    self.emit_log('info', '🔥 SmartCash model prepared for Phase 2 (backbone unfrozen)')
                else:
                    # Use legacy approach for other models
                    self.model = ModelUtils.rebuild_model_for_phase2(
                        self.model_api, self.model, config
                    )
                    self.emit_log('info', '🔥 Model prepared for Phase 2 (backbone unfrozen)')
            else:
                # Phase 1 uses model as-is (backbone frozen)
                if is_smartcash_model:
                    # Ensure SmartCash model is in Phase 1
                    self.model.current_phase = 1
                    self.emit_log('info', '❄️ SmartCash model prepared for Phase 1 (backbone frozen)')
                else:
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
            
            info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),
                'architecture': self.model.__class__.__name__,
                'device': str(next(self.model.parameters()).device),
                'memory_optimizer_available': self.memory_optimizer is not None
            }
            
            # Add SmartCash-specific information
            if isinstance(self.model, SmartCashYOLOv5Model):
                phase_info = self.model.get_phase_info()
                model_config = self.model.get_model_config()
                
                info.update({
                    'backbone': model_config.get('backbone', 'unknown'),
                    'num_classes': model_config.get('num_classes', 17),
                    'img_size': model_config.get('img_size', 640),
                    'current_phase': phase_info.get('phase', 1),
                    'backbone_frozen': phase_info.get('backbone_frozen', True),
                    'trainable_ratio': phase_info.get('trainable_ratio', 0.0),
                    'class_mapping': model_config.get('class_mapping', {})
                })
            
            return info
            
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