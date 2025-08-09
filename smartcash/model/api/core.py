"""
SmartCash Model API with YOLOv5 Integration
Multi-Layer Banknote Detection System
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import yaml
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.model.core.model_builder import ModelBuilder
from smartcash.model.core.checkpoints.checkpoint_manager import CheckpointManager
from smartcash.model.utils.device_utils import setup_device, get_device_info
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
from smartcash.model.architectures.model import SmartCashYOLOv5Model



class SmartCashModelAPI:
    """
    SmartCash Model API with YOLOv5 integration support
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 progress_callback: Optional[Callable] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize SmartCash Model API
        
        Args:
            config_path: Path to configuration file
            progress_callback: Progress callback function
            config: Configuration dictionary
        """
        self.logger = get_logger("model.api")
        self.progress_bridge = TrainingProgressTracker(progress_callback, True, 'single_phase')
        
        # Load configuration
        if config is not None:
            self.config = config
            self.logger.debug("📋 Using provided configuration dictionary")
        else:
            self.config = self._load_config(config_path)
        
        self.device = setup_device(self.config.get('device', {}))
        
        # Initialize memory optimizer
        self.memory_optimizer = get_memory_optimizer(self.device)
        memory_config = self.memory_optimizer.setup_memory_efficient_settings()
        
        # Initialize model builder and checkpoint manager
        self.model_builder = ModelBuilder(self.config, self.progress_bridge)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Model state
        self.model = None
        self.is_model_built = False
        
        # Store memory configuration
        self.memory_config = memory_config
        
        self.logger.debug("✅ SmartCash Model API initialized")
        
        # Log available architectures
        available_archs = self.model_builder.get_available_architectures()
        self.logger.debug(f"🏗️ Available architectures: {available_archs}")
        
        # Log device info
        device_info = get_device_info()
        self.logger.info(f"🔧 Using device: {device_info['device']}")
        if device_info['device'] == 'cuda':
            self.logger.info(f"   GPU: {device_info['gpu_name']}, Memory: {device_info['gpu_memory']}MB")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"📋 Configuration loaded from {config_path}")
        else:
            # Default configuration
            config = {
                'device': {'type': 'auto'},
                'model': {
                    'backbone': 'cspdarknet',
                    'num_classes': 7,
                    'img_size': 640
                },
                'training': {
                    'batch_size': 16,
                    'epochs': 100,
                    'learning_rate': 1e-3
                }
            }
            self.logger.info("📋 Using default configuration")
        
        return config
    
    def build_model(self, model_config: Dict[str, Any] = None, use_smartcash_architecture: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Build model using enhanced builder or new SmartCash architecture
        
        Args:
            model_config: Model configuration dictionary
            use_smartcash_architecture: Whether to use new SmartCashYOLOv5Model directly
            **kwargs: Additional model parameters
            
        Returns:
            Build result dictionary
        """
        try:
            self.progress_bridge.start_operation("build_model", total_steps=4)
            
            # Merge configuration
            if model_config is None:
                model_config = self.config.get('model', {})
            
            # Override with kwargs
            model_config.update(kwargs)
            
            # CRITICAL FIX: Update API's internal config with the actual model config being used
            if 'model' not in self.config:
                self.config['model'] = {}
            self.config['model'].update(model_config)
            
            if use_smartcash_architecture:
                # Use new SmartCashYOLOv5Model directly
                result = self._build_smartcash_model(model_config)
            else:
                # Use legacy model builder
                result = self._build_legacy_model(model_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model building failed: {str(e)}")
            self.progress_bridge.error(f"Model building failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'model': None
            }
    
    def _build_smartcash_model(self, config: Dict[str, Any]) -> bool:
        """
        Build the SmartCash model with the given configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            bool: True if model was built successfully, False otherwise
        """
        try:
            self.progress_bridge.start_operation("build_model", 4)  # 4 steps
            
            # Initialize model builder if not already done
            if not hasattr(self, 'model_builder'):
                self.model_builder = ModelBuilder(
                    config=config,
                    device=self.device,
                    logger=self.logger
                )
            
            # Build the model
            self.progress_bridge.update_substep(1, 4, "Initializing model architecture...")
            self.model = self.model_builder.build()
            
            if self.model is None:
                error_msg = "Failed to build model: Model builder returned None"
                self.logger.error(error_msg)
                self.progress_bridge.error(error_msg)
                return False
            
            self.progress_bridge.update_substep(2, 4, "Model architecture initialized")
            
            # Move model to device
            self.progress_bridge.update_substep(3, 4, f"Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            self.progress_bridge.update_substep(4, 4, f"Model moved to {self.device}")
            
            # Initialize checkpoint manager with proper configuration
            checkpoint_config = {
                'checkpoint': {
                    'save_dir': config.get('checkpoint_dir', 'checkpoints'),
                    'max_checkpoints': config.get('max_checkpoints', 5),
                    'auto_cleanup': True
                },
                'model': {
                    'backbone': config.get('backbone', 'yolov5s')
                }
            }
            
            self.checkpoint_manager = CheckpointManager(
                config=checkpoint_config,
                is_resuming=config.get('resume_training', False)
            )
            
            self.progress_bridge.complete_operation(4, "Model built successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to build model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.progress_bridge.error(error_msg)
            return False
    
    def build_model(self, **kwargs) -> Dict[str, Any]:
        """
        Build the model with the given configuration.
        
        Args:
            **kwargs: Model configuration parameters
            
        Returns:
            Dict containing build status and model info
        """
        try:
            # Update config with any provided parameters
            if kwargs:
                self.config.update(kwargs)
            
            # Build the model
            success = self._build_smartcash_model(self.config)
            
            if not success or self.model is None:
                return {
                    'success': False,
                    'error': 'Failed to build model',
                    'model': None
                }
            
            # Get model info
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'success': True,
                'model': str(self.model),
                'parameters': total_params,
                'device': str(self.device)
            }
            
        except Exception as e:
            error_msg = f"Error building model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'model': None
            }
            
    def train_model(self, train_data: str, val_data: str, epochs: int, 
                   batch_size: int, learning_rate: float, checkpoint_dir: str, 
                   log_dir: str, progress_callback: Optional[Callable] = None, 
                   **kwargs) -> Dict[str, Any]:
        """
        Train the model with the given parameters.
        
        Args:
            train_data: Path to training data
            val_data: Path to validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            progress_callback: Callback for training progress updates
            **kwargs: Additional training parameters
            
        Returns:
            Dict containing training results
        """
        from smartcash.model.training.training_pipeline import TrainingPipeline
        
        try:
            # Update config with training parameters
            self.config.update({
                'train_data': train_data,
                'val_data': val_data,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'checkpoint_dir': checkpoint_dir,
                'log_dir': log_dir,
                **kwargs
            })
            
            # Initialize training pipeline
            training_pipeline = TrainingPipeline(
                progress_callback=progress_callback,
                verbose=kwargs.get('verbose', True)
            )
            
            # Start training
            self.logger.info("🚀 Starting model training...")
            
            # Update config with model and device info
            training_config = self.config.copy()
            training_config.update({
                'model': self.model,
                'device': self.device,
                'checkpoint_manager': self.checkpoint_manager,
                'train_data': train_data,
                'val_data': val_data,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'checkpoint_dir': checkpoint_dir,
                'log_dir': log_dir,
                **kwargs
            })
            
            # Call run_training with the updated config
            result = training_pipeline.run_training(
                epochs=epochs,
                config=training_config
            )
            
            if result.get('success', False):
                self.logger.info("✅ Training completed successfully")
            else:
                self.logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'exception': str(e)
            }
    
    def _get_smartcash_model_info(self) -> Dict[str, Any]:
        """Get model information for SmartCash model."""
        if not isinstance(self.model, SmartCashYOLOv5Model):
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        phase_info = self.model.get_phase_info()
        model_config = self.model.get_model_config()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'backbone': model_config.get('backbone'),
            'num_classes': model_config.get('num_classes'),
            'img_size': model_config.get('img_size'),
            'current_phase': phase_info.get('phase'),
            'backbone_frozen': phase_info.get('backbone_frozen'),
            'trainable_ratio': phase_info.get('trainable_ratio')
        }
    
    def get_available_architectures(self) -> List[str]:
        """Get list of available architecture types"""
        return self.model_builder.get_available_architectures()
    
    def get_available_backbones(self) -> List[str]:
        """Get list of available backbones for YOLOv5 architecture"""
        return self.model_builder.get_available_backbones('yolov5')
    
    def validate_model(self, input_size: tuple = (1, 3, 640, 640)) -> Dict[str, Any]:
        """
        Validate model with dummy input
        
        Args:
            input_size: Input tensor size for validation
            
        Returns:
            Validation result dictionary
        """
        if not self.is_model_built:
            return {'success': False, 'error': 'Model not built yet'}
        
        try:
            # Handle SmartCash models
            if isinstance(self.model, SmartCashYOLOv5Model):
                return self._validate_smartcash_model(input_size)
            
            # Get the actual model from the wrapper if it exists (legacy)
            model_to_validate = self.model.yolov5_model if hasattr(self.model, 'yolov5_model') else self.model
            model_to_validate.eval()
            dummy_input = torch.randn(input_size).to(self.device)
            
            with torch.no_grad():
                output = model_to_validate(dummy_input)
            
            # Analyze output structure
            output_info = self._analyze_model_output(output)
            
            return {
                'success': True,
                'input_shape': input_size,
                'output_info': output_info,
                'device': str(self.device),
                'model_mode': 'eval'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_smartcash_model(self, input_size: tuple = (1, 3, 640, 640)) -> Dict[str, Any]:
        """Validate SmartCash model specifically."""
        try:
            self.model.eval()
            dummy_input = torch.randn(input_size).to(self.device)
            
            with torch.no_grad():
                # Test training mode output
                training_output = self.model(dummy_input, training=True)
                # Test inference mode output  
                inference_output = self.model(dummy_input, training=False)
            
            # Get model configuration
            model_config = self.model.get_model_config()
            phase_info = self.model.get_phase_info()
            
            return {
                'success': True,
                'input_shape': input_size,
                'output_info': {
                    'training_output_type': type(training_output).__name__,
                    'inference_output_type': type(inference_output).__name__,
                    'training_shape': getattr(training_output, 'shape', 'N/A'),
                    'inference_length': len(inference_output) if isinstance(inference_output, list) else 'N/A'
                },
                'model_config': model_config,
                'phase_info': phase_info,
                'device': str(self.device),
                'model_mode': 'SmartCashYOLOv5Model'
            }
            
        except Exception as e:
            self.logger.error(f"❌ SmartCash model validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_model_output(self, output) -> Dict[str, Any]:
        """Analyze model output structure"""
        if isinstance(output, dict):
            # Multi-layer output
            layer_info = {}
            for layer_name, layer_output in output.items():
                if isinstance(layer_output, (list, tuple)):
                    layer_info[layer_name] = {
                        'type': 'list',
                        'length': len(layer_output),
                        'shapes': [x.shape if hasattr(x, 'shape') else str(type(x)) for x in layer_output]
                    }
                else:
                    layer_info[layer_name] = {
                        'type': 'tensor',
                        'shape': output.shape if hasattr(output, 'shape') else str(type(output))
                    }
            
            return {
                'output_type': 'multi_layer',
                'layers': layer_info,
                'num_layers': len(output)
            }
        
        elif isinstance(output, (list, tuple)):
            # Single layer with multiple scales
            return {
                'output_type': 'single_layer_multi_scale',
                'num_scales': len(output),
                'shapes': [x.shape if hasattr(x, 'shape') else str(type(x)) for x in output]
            }
        
        else:
            # Single tensor output
            return {
                'output_type': 'single_tensor',
                'shape': output.shape if hasattr(output, 'shape') else str(type(output))
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if not self.is_model_built:
            return {'error': 'Model not built yet'}
        
        summary = {
            'is_built': self.is_model_built,
            'architecture_type': 'yolov5',
            'device': str(self.device)
        }
        
        # Get model-specific info
        model_info = self.model_builder.get_model_info(self.model)
        summary.update(model_info)
        
        # Get device info
        device_info = get_device_info()
        summary['device_info'] = device_info
        
        return summary
    
    def save_model(self, checkpoint_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Save model checkpoint"""
        if not self.is_model_built:
            return {'success': False, 'error': 'Model not built yet'}
        
        try:
            # Add architecture info to metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'architecture_type': 'yolov5',
                'build_timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            })
            
            # Use checkpoint manager
            result = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                checkpoint_path=checkpoint_path,
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_checkpoint(self, **kwargs) -> str:
        """
        Save model checkpoint with enhanced metadata.
        
        Args:
            **kwargs: Checkpoint parameters including:
                - epoch: Current epoch number
                - phase: Training phase number
                - metrics: Training metrics dictionary
                - is_best: Whether this is the best checkpoint
                - config: Configuration dictionary
                - session_id: Training session ID (optional)
                
        Returns:
            Path to saved checkpoint
        """
        if not self.is_model_built:
            raise RuntimeError("Model not built yet")
        
        try:
            # Extract parameters
            epoch = kwargs.get('epoch', 0)
            phase = kwargs.get('phase', 1)
            metrics = kwargs.get('metrics', {})
            is_best = kwargs.get('is_best', False)
            config = kwargs.get('config', self.config)
            session_id = kwargs.get('session_id')
            
            # Prepare metadata
            metadata = {
                'epoch': epoch,
                'phase': phase,
                'metrics': metrics,
                'is_best': is_best,
                'config': config,
                'session_id': session_id,
                'architecture_type': 'yolov5',
                'build_timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            }
            
            # Generate checkpoint name if not provided
            checkpoint_name = kwargs.get('checkpoint_name')
            if not checkpoint_name:
                # Create a copy of kwargs without metrics to avoid duplicate argument error
                name_kwargs = {k: v for k, v in kwargs.items() if k != 'metrics'}
                checkpoint_name = self.checkpoint_manager._generate_checkpoint_name(metrics, **name_kwargs)
            
            # Save checkpoint using checkpoint manager
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                metrics=metrics,
                checkpoint_name=checkpoint_name,
                epoch=epoch,
                phase=phase,
                is_best=is_best,
                config=config,
                session_id=session_id,
                model_api=self  # CRITICAL FIX: Pass model API for proper config extraction
            )
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint saving failed: {str(e)}")
            raise RuntimeError(f"Checkpoint saving failed: {str(e)}")
    
    def load_model(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model from checkpoint"""
        try:
            result = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            if result.get('success'):
                self.model = result['model'].to(self.device)
                self.is_model_built = True
                
                # Set architecture type for loaded model
                self.logger.info(f"✅ Model loaded: yolov5")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional loading parameters
            
        Returns:
            Loading result dictionary
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                # Build model first if not already built
                build_result = self.build_model(**kwargs)
                if not build_result.get('success', False):
                    return build_result
            
            result = self.checkpoint_manager.load_checkpoint(self.model, checkpoint_path)
            self.logger.info(f"✅ Checkpoint loaded: {result['checkpoint_path']}")
            
            # Add success field for evaluation service compatibility
            result['success'] = True
            return result
            
        except Exception as e:
            error_msg = f"❌ Checkpoint loading failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': str(e)
            }


def create_api(config_path: Optional[str] = None, **kwargs) -> 'SmartCashModelAPI':
    """
    Create a new SmartCash Model API instance
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        SmartCashModelAPI instance
    """
    return SmartCashModelAPI(config_path=config_path, **kwargs)


def run_full_training_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Run full training pipeline with simplified interface for examples.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Dictionary containing training results
    """
    from smartcash.model.training.training_pipeline import TrainingPipeline
    from smartcash.model.training.pipeline.configuration_builder import ConfigurationBuilder
    import uuid
    
    try:
        # Extract callbacks
        progress_callback = kwargs.pop('progress_callback', None)
        log_callback = kwargs.pop('log_callback', None)
        metrics_callback = kwargs.pop('metrics_callback', None)
        verbose = kwargs.pop('verbose', True)
        
        # Create training pipeline
        pipeline = TrainingPipeline(
            progress_callback=progress_callback,
            log_callback=log_callback,
            metrics_callback=metrics_callback,
            verbose=verbose
        )
        
        # Create session ID if not provided
        session_id = kwargs.get('session_id', str(uuid.uuid4())[:8])
        
        # CRITICAL FIX: Use ConfigurationBuilder to create structured config
        config_builder = ConfigurationBuilder(session_id)
        structured_config = config_builder.build_training_config(**kwargs)
        
        # Extract training parameters for the pipeline interface
        total_epochs = structured_config.get('phase_1_epochs', 10) + structured_config.get('phase_2_epochs', 0)
        
        # Run training with properly structured configuration
        result = pipeline.run_training(
            epochs=total_epochs,
            config=structured_config  # Use structured config instead of flat kwargs
        )
        
        return result
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Full training pipeline failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


# Export key classes and functions
__all__ = [
    'SmartCashModelAPI',
    'create_api',
    'run_full_training_pipeline'
]