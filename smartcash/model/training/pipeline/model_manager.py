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

    def setup_model(self, config: Dict[str, Any], use_yolov5_integration: bool = True, use_smartcash_architecture: bool = True, is_resuming: bool = False) -> Tuple[Any, Any]:
        """
        Set up model API and build model according to configuration.
        
        Args:
            config: Training configuration
            use_yolov5_integration: Whether to use YOLOv5 integration (legacy)
            use_smartcash_architecture: Whether to use new SmartCashYOLOv5Model directly
            is_resuming: Whether this is a resume operation
            
        Returns:
            Tuple of (model_api, built_model)
        """
        try:
            self.emit_log('info', '🏗️ Setting up model for training pipeline')
            
            # Initialize checkpoint manager here, where config is available
            self.checkpoint_manager = create_checkpoint_manager(config, is_resuming=is_resuming)

            if use_smartcash_architecture:
                # Use new SmartCashYOLOv5Model directly
                self.emit_log('info', '🆕 Using SmartCashYOLOv5Model architecture')
                self.model = self._create_smartcash_model_directly(config)
                
                # For SmartCash model, we create a lightweight API wrapper
                self.model_api = self._create_smartcash_api_wrapper(config)
                
                # Move model to appropriate device
                device = next(self.model.parameters()).device
                self.model = self.model.to(device)
                
            else:
                # Use legacy approach for backward compatibility
                self.emit_log('info', '🔄 Using legacy model API approach')
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
    
    def _create_smartcash_model_directly(self, config: Dict[str, Any]) -> SmartCashYOLOv5Model:
        """Create SmartCashYOLOv5Model directly without the model API wrapper."""
        try:
            model_config = config.get('model', {})
            
            # Extract parameters for SmartCashYOLOv5Model
            backbone = model_config.get('backbone', 'yolov5s')
            
            # CRITICAL FIX: SmartCash architecture always uses 17 classes (never the legacy dict format)
            raw_num_classes = model_config.get('num_classes', 17)
            if isinstance(raw_num_classes, dict):
                # Legacy format detected - force to 17 for SmartCash
                num_classes = 17
                self.emit_log('info', '🔧 Fixed SmartCash num_classes: dict format overridden to 17', {})
            else:
                num_classes = raw_num_classes
            img_size = model_config.get('img_size', 640)
            pretrained = model_config.get('pretrained', True)
            device = config.get('device', {}).get('type', 'auto')
            
            
            # Create the SmartCash YOLOv5 model
            model = SmartCashYOLOv5Model(
                backbone=backbone,
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                device=device
            )
            
            self.emit_log('info', '✅ SmartCashYOLOv5Model created directly', {
                'backbone': backbone,
                'num_classes': num_classes,
                'img_size': img_size,
                'pretrained': pretrained,
                'device': device
            })
            
            return model
            
        except Exception as e:
            self.emit_log('error', f'❌ Failed to create SmartCashYOLOv5Model: {str(e)}')
            raise
    
    def _create_smartcash_api_wrapper(self, config: Dict[str, Any]) -> Any:
        """Create a lightweight API wrapper for SmartCashYOLOv5Model."""
        
        class SmartCashAPIWrapper:
            """Lightweight API wrapper for SmartCashYOLOv5Model to maintain interface compatibility."""
            
            def __init__(self, model: SmartCashYOLOv5Model, config: Dict[str, Any]):
                self.model = model
                self.config = config
                self.is_model_built = True
                
            def get_backbone_name(self) -> str:
                """Get backbone name from the model."""
                return self.model.backbone_type
                
            def validate_model(self) -> Dict[str, Any]:
                """Validate the model."""
                try:
                    # Simple validation by checking if model can process dummy input
                    dummy_input = torch.randn(1, 3, 640, 640).to(self.model.device)
                    with torch.no_grad():
                        output = self.model(dummy_input)
                    return {
                        'success': True,
                        'output_type': 'smartcash_yolov5',
                        'backbone': self.model.backbone_type,
                        'num_classes': self.model.num_classes
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e)
                    }
            
            def save_checkpoint(self, **kwargs) -> str:
                """Save checkpoint using the model's own configuration."""
                # Get model configuration for checkpoint
                model_config = self.model.get_model_config()
                
                # Basic checkpoint structure for SmartCash model
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'model_config': model_config,
                    'architecture': 'SmartCashYOLOv5Model',
                    **kwargs
                }
                
                # Generate checkpoint path if not provided
                checkpoint_path = kwargs.get('checkpoint_path', 'data/checkpoints/smartcash_checkpoint.pt')
                torch.save(checkpoint, checkpoint_path)
                
                return checkpoint_path
        
        return SmartCashAPIWrapper(self.model, config)
    
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