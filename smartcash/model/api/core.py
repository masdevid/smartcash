"""
SmartCash Model API with YOLOv5 Integration
Multi-Layer Banknote Detection System
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.model.core.model_builder import ModelBuilder
from smartcash.model.core.checkpoint_manager import CheckpointManager
from smartcash.model.utils.device_utils import setup_device, get_device_info
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker
from smartcash.model.utils.memory_optimizer import get_memory_optimizer



class SmartCashModelAPI:
    """
    SmartCash Model API with YOLOv5 integration support
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 progress_callback: Optional[Callable] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 use_yolov5_integration: bool = True):
        """
        Initialize SmartCash API with YOLOv5 integration
        
        Args:
            config_path: Path to configuration file
            progress_callback: Progress callback function
            config: Configuration dictionary
            use_yolov5_integration: Enable YOLOv5 integration
        """
        self.logger = get_logger("model.api")
        self.progress_bridge = TrainingProgressTracker(progress_callback, True, 'single_phase')
        self.use_yolov5_integration = use_yolov5_integration
        
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
        
        # Initialize YOLOv5-compatible model builder
        self.model_builder = ModelBuilder(self.config, self.progress_bridge)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        
        # Model state
        self.model = None
        self.is_model_built = False
        
        # Store memory configuration
        self.memory_config = memory_config
        
        self.logger.debug("✅ SmartCash Model API initialized")
        self.logger.debug(f"🔧 YOLOv5 integration: {'enabled' if use_yolov5_integration else 'disabled'}")
        
        # Log available architectures
        available_archs = self.model_builder.get_available_architectures()
        self.logger.debug(f"🏗️ Available architectures: {available_archs}")
    
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
    
    def build_model(self, model_config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Build model using enhanced builder
        
        Args:
            model_config: Model configuration dictionary
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
            
            # Set defaults
            backbone = model_config.get('backbone', 'cspdarknet')
            
            self.logger.debug(f"🔧 Building model: {backbone} | Architecture: yolov5")
            
            # Build model
            self.model = self.model_builder.build(
                backbone=backbone,
                detection_layers=model_config.get('detection_layers', ['layer_1', 'layer_2', 'layer_3']),
                layer_mode=model_config.get('layer_mode', 'multi'),
                num_classes=model_config.get('num_classes', 7),
                img_size=model_config.get('img_size', 640),
                pretrained=model_config.get('pretrained', True),
                feature_optimization=model_config.get('feature_optimization', {'enabled': True})
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.is_model_built = True
            
            # Get model info
            model_info = self.model_builder.get_model_info(self.model)
            
            self.progress_bridge.complete(4, "Model build complete")
            
            result = {
                'success': True,
                'model': self.model,
                'model_info': model_info,
                'architecture_type': 'yolov5',
                'device': str(self.device),
                'memory_config': self.memory_config
            }
            
            self.logger.info(f"✅ Model built successfully: {model_info.get('total_parameters', 0):,} parameters")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model building failed: {str(e)}")
            self.progress_bridge.error(f"Model building failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'model': None
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
            self.model.eval()
            dummy_input = torch.randn(input_size).to(self.device)
            
            with torch.no_grad():
                output = self.model(dummy_input)
            
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


def create_api(config_path: Optional[str] = None, **kwargs) -> SmartCashModelAPI:
    """
    Convenience function to create SmartCash API instance
    
    Args:
        config_path: Configuration file path
        **kwargs: Additional API parameters
        
    Returns:
        SmartCashModelAPI instance
    """
    return SmartCashModelAPI(config_path=config_path, **kwargs)


# Training pipeline function for backwards compatibility
def run_full_training_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Run full training pipeline using SmartCash YOLOv5 architecture
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Training result dictionary
    """
    from smartcash.model.training.training_pipeline import TrainingPipeline
    
    # Extract pipeline constructor arguments
    pipeline_args = {
        'progress_callback': kwargs.pop('progress_callback', None),
        'log_callback': kwargs.pop('log_callback', None),
        'live_chart_callback': kwargs.pop('live_chart_callback', None),
        'metrics_callback': kwargs.pop('metrics_callback', None),
        'verbose': kwargs.pop('verbose', True),
        'use_yolov5_integration': kwargs.pop('use_yolov5_integration', True)
    }
    
    # Create training pipeline
    pipeline = TrainingPipeline(**pipeline_args)
    
    # Run training
    return pipeline.run_full_training_pipeline(patience=kwargs.pop('patience', 30), **kwargs)


# Backwards compatibility aliases
def create_model_api(config_path: Optional[str] = None, **kwargs) -> SmartCashModelAPI:
    """Backwards compatibility function"""
    return SmartCashModelAPI(config_path=config_path, **kwargs)


def quick_build_model(backbone: str = 'cspdarknet', **kwargs):
    """Quick model building function for backwards compatibility"""
    api = SmartCashModelAPI()
    return api.build_model({'backbone': backbone, **kwargs})


# Export key classes and functions
__all__ = [
    'SmartCashModelAPI',
    'create_api',
    'create_model_api',
    'quick_build_model', 
    'run_full_training_pipeline'
]