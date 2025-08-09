"""
SmartCash Model Builder with YOLOv5 Integration
Multi-Layer Banknote Detection System

This module provides a model builder that creates and configures YOLOv5-based models
for the SmartCash system, supporting various backbones and configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker

# Import YOLOv5 model factory
try:
    from smartcash.model.architectures.yolov5.model_factory import YOLOv5ModelFactory
    YOLOV5_AVAILABLE = True
except ImportError as e:
    get_logger("model.builder").warning(f"YOLOv5 integration not available: {e}")
    YOLOV5_AVAILABLE = False


class ModelBuilder:
    """
    SmartCash model builder with YOLOv5 integration for multi-layer banknote detection.
    
    This class handles the creation and configuration of models for the SmartCash system,
    supporting various backbones and training configurations.
    """
    
    # Class variable to track YOLOv5 availability
    YOLOV5_AVAILABLE = YOLOV5_AVAILABLE
    
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
    
    def build(self, backbone: str = 'yolov5s', num_classes: int = 17, 
              img_size: int = 640, pretrained: bool = True, **kwargs) -> nn.Module:
        """
        Build a YOLOv5-based SmartCash model.
        
        Args:
            backbone: Backbone type ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'efficientnet_b4')
            num_classes: Number of output classes (default: 17 for SmartCash)
            img_size: Input image size (must be multiple of 32, default: 640)
            pretrained: Whether to use pretrained weights (default: True)
            **kwargs: Additional model configuration options:
                - device: Device to place model on ('auto', 'cuda', 'mps', 'cpu')
                - freeze: List of layer names to freeze (e.g., ['backbone', 'neck'])
                
        Returns:
            nn.Module: Configured YOLOv5 model instance
            
        Raises:
            RuntimeError: If model creation fails or YOLOv5 is not available
        """
        if not self.YOLOV5_AVAILABLE or self.model_factory is None:
            error_msg = ("❌ YOLOv5 integration is required but not available. "
                        "Please ensure YOLOv5 is properly installed.")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log model creation
        if self.progress_bridge:
            self.progress_bridge.update_operation(1, 3, f"Building YOLOv5 model: {backbone}")
        self.logger.info(f"🏗️ Building YOLOv5 model with backbone: {backbone}, "
                        f"classes: {num_classes}, img_size: {img_size}")
        
        try:
            # Create model using the factory
            model = self.model_factory.create_model(
                backbone=backbone,
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                **kwargs
            )
            
            # Log successful model creation
            model_info = self.get_model_info(model)
            param_count = model_info.get('parameters', 0)
            self.logger.info(f"✅ Model built successfully: {param_count:,} parameters")
            
            return model
            
        except Exception as e:
            error_msg = f"❌ Failed to build YOLOv5 model: {str(e)}"
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
                img_size: int = 640, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Create a SmartCash YOLOv5 model with the specified configuration.
    
    This is a convenience wrapper around ModelBuilder that provides a simple interface
    for creating pre-configured YOLOv5 models for SmartCash.
    
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
        nn.Module: Configured YOLOv5 model ready for training or inference.
        
    Example:
        >>> # Create a small YOLOv5 model
        >>> model = create_model('yolov5s', num_classes=17)
        >>> 
        >>> # Create a larger model with custom image size
        >>> model = create_model('yolov5m', num_classes=17, img_size=1280)
    """
    # Create model builder with default config
    builder = ModelBuilder(config={})
    
    # Build and return the model
    return builder.build(
        backbone=backbone,
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        **kwargs
    )


# Export key classes and functions
__all__ = [
    'ModelBuilder',
    'create_model'
]