"""
SmartCash Model Builder
Multi-Layer Banknote Detection System

This module provides a model builder that creates and configures models
for the SmartCash system, supporting various backbones and configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker
from smartcash.model.architectures.model import SmartCashYOLOv5Model


class ModelBuilder:
    """
    SmartCash model builder with YOLOv5 integration for multi-layer banknote detection.
    
    This class handles the creation and configuration of models for the SmartCash system,
    supporting various backbones and training configurations.
    """
    
    # Class variable for backward compatibility
    YOLOV5_AVAILABLE = True
    
    def __init__(self, config: Dict[str, Any], progress_bridge: TrainingProgressTracker = None):
        """
        Initialize SmartCash model builder with YOLOv5 integration
        
        Args:
            config: Configuration dictionary
            progress_bridge: Progress tracking bridge for reporting progress
        """
        self.config = config
        self.progress_bridge = progress_bridge
        self.logger = get_logger("model.builder") or get_logger("smartcash")
        
        # Set device based on availability
        self.device = self._setup_device()
        
        # Initialize YOLOv5 model factory if available
        self.model_factory = None
        if self.YOLOV5_AVAILABLE:
            try:
                self.model_factory = YOLOv5ModelFactory(logger=self.logger)
                self.logger.info("✅ YOLOv5 model factory initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize YOLOv5 model factory: {e}")
                self.YOLOV5_AVAILABLE = False
        
        if not self.YOLOV5_AVAILABLE:
            self.logger.warning("⚠️ YOLOv5 integration not available. Some features may be limited.")
    
    def _setup_device(self) -> torch.device:
        """Set up the device for model training/inference."""
        # Check for CUDA first
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Then check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("Using MPS (Metal Performance Shaders) on Apple Silicon")
        # Fall back to CPU
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU (no CUDA or MPS available)")
        
        # Set environment variables for optimal performance
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'
        
        return device
    
    def build(self, model_cfg: Optional[Dict[str, Any]] = None, device: str = 'auto') -> nn.Module:
        """
        Build a SmartCash model based on the configuration.
        
        Args:
            model_cfg: Model configuration (uses self.config if None)
            device: Device to place model on ('auto', 'cuda', 'mps', 'cpu')
            
        Returns:
            Configured SmartCash model
        """
        cfg = model_cfg or self.config.get('model', {})
        
        # Get model configuration with defaults
        backbone = cfg.get('backbone', 'yolov5s')
        num_classes = cfg.get('num_classes', 17)
        img_size = cfg.get('img_size', 640)
        pretrained = cfg.get('pretrained', True)
        
        # Log model creation
        if self.progress_bridge:
            self.progress_bridge.update_operation(1, 3, f"Building model: {backbone}")
            
        self.logger.info(f"Building model with backbone: {backbone}, "
                       f"classes: {num_classes}, img_size: {img_size}")
        
        try:
            # Create model using the new SmartCashYOLOv5Model
            model = SmartCashYOLOv5Model(
                backbone=backbone,
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                device=device
            )
            
            # Log successful model creation
            model_info = self.get_model_info(model)
            param_count = model_info.get('parameters', 0)
            self.logger.info(f"✅ Model built successfully: {param_count:,} parameters")
            
            return model
            
        except Exception as e:
            error_msg = f"❌ Failed to build model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _optimize_model_features(self, model: nn.Module, optimization_config: Dict) -> nn.Module:
        """
        Apply feature optimization to the model.
        
        Args:
            model: The model to optimize
            optimization_config: Optimization configuration
            
        Returns:
            Optimized model
        """
        if not optimization_config.get('enabled', True):
            return model
            
        self.logger.info("🔧 Applying feature optimization...")
        
        try:
            # Apply optimizations if the model supports them
            if hasattr(model, 'freeze_backbone'):
                model.freeze_backbone()
                self.logger.info("  - Frozen backbone layers")
                
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                self.logger.info("  - Enabled gradient checkpointing")
                
            # Apply custom learning rates if specified
            custom_lr = optimization_config.get('custom_lr')
            if custom_lr and hasattr(model, 'set_parameter_lrs'):
                model.set_parameter_lrs(custom_lr)
                self.logger.info("  - Applied custom learning rates")
                
        except Exception as e:
            self.logger.warning(f"Feature optimization partially failed: {e}")
            
        return model
        
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_available_architectures(self) -> List[str]:
        """Get list of available architecture types."""
        return ['yolov5']  # Only YOLOv5 is supported
    
    def get_available_backbones(self) -> List[str]:
        """Get list of available backbones for the architecture."""
        return ['cspdarknet', 'efficientnet_b4']  # Supported backbones
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: The model to get info for
            
        Returns:
            Dictionary containing model information including:
            - parameters: Number of trainable parameters
            - device: Device the model is on
            - class_names: List of class names (if available)
            - model_stride: Model stride (if available)
        """
        info = {
            'parameters': self._count_parameters(model),
            'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') and list(model.parameters()) else 'unknown'
        }
        
        # Get YOLOv5 model information if available
        try:
            if hasattr(model, 'yolov5_model'):
                yolo_model = model.yolov5_model
                
                # Extract model information using the model factory if available
                if self.model_factory and hasattr(self.model_factory, 'get_model_info'):
                    yolo_info = self.model_factory.get_model_info(yolo_model)
                    info.update(yolo_info)
                else:
                    # Fallback to direct attribute access
                    if hasattr(yolo_model, 'names'):
                        info['class_names'] = yolo_model.names
                    if hasattr(yolo_model, 'stride'):
                        info['model_stride'] = yolo_model.stride.tolist() \
                            if hasattr(yolo_model.stride, 'tolist') else str(yolo_model.stride)
                            
        except Exception as e:
            self.logger.debug(f"Could not extract YOLOv5 model info: {e}")
        
        return info


def create_model(backbone: str = 'yolov5s', num_classes: int = 17, 
                img_size: int = 640, pretrained: bool = True, **kwargs):
    """
    Create a SmartCash model with the specified configuration.
    
    This is a convenience wrapper around ModelBuilder that provides a simple interface
    for creating pre-configured models for SmartCash.
    
    Args:
        backbone: Backbone architecture. One of: 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'efficientnet_b4'.
                 Default: 'yolov5s' (smallest and fastest).
        num_classes: Number of output classes. Default: 17 (for SmartCash).
        img_size: Input image size (height=width). Must be multiple of 32. Default: 640.
        pretrained: Whether to load pretrained weights. Default: True.
        **kwargs: Additional arguments passed to ModelBuilder.build():
            - device: Device to place model on ('auto', 'cuda', 'mps', 'cpu')
            - freeze: List of layer names to freeze (e.g., ['backbone', 'neck'])
            
    Returns:
        nn.Module: Configured model ready for training or inference.
        
    Example:
        >>> # Create a small model
        >>> model = create_model('yolov5s', num_classes=17)
        >>> 
        >>> # Create a larger model with custom image size
        >>> model = create_model('yolov5m', num_classes=17, img_size=1280)
    """
    # Create a minimal config dictionary
    config = {
        'model': {
            'backbone': backbone,
            'num_classes': num_classes,
            'img_size': img_size,
            'pretrained': pretrained
        }
    }
    
    # Get device from kwargs or use 'auto'
    device = kwargs.get('device', 'auto')
    
    # Create model builder and build the model
    builder = ModelBuilder(config=config)
    return builder.build(device=device)


# Export key classes and functions
__all__ = [
    'ModelBuilder',
    'create_model'
]