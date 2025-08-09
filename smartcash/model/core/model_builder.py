"""
SmartCash Model Builder with YOLOv5 Integration
Multi-Layer Banknote Detection System

This module provides a model builder that creates and configures models for the SmartCash system.
It supports both training and inference with various backbones and configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker

# Import the new modular YOLOv5 integration
try:
    from smartcash.model.architectures.yolov5 import (
        SmartCashYOLOv5Integration,
        create_training_model,
        create_smartcash_yolov5_model,
        YOLOv5ModelFactory
    )
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
    
    def build(self, backbone: str = 'efficientnet_b4', detection_layers: List[str] = None,
              layer_mode: str = 'multi', num_classes: int = 7, img_size: int = 640,
              feature_optimization: Dict = None, pretrained: bool = True, **kwargs) -> nn.Module:
        """
        Build model with YOLOv5 architecture
        
        Args:
            backbone: Backbone type ('cspdarknet', 'efficientnet_b4')
            detection_layers: Detection layers to use
            layer_mode: 'single', 'multi' for layer mode
            num_classes: Number of classes for primary detection layer
            img_size: Input image size
            feature_optimization: Feature optimization settings
            pretrained: Use pretrained weights
            **kwargs: Additional arguments
            
        Returns:
            YOLOv5 integrated model instance
        """
        detection_layers = detection_layers or ['layer_1', 'layer_2', 'layer_3']
        feature_optimization = feature_optimization or {'enabled': True}
        
        if self.progress_bridge:
            self.progress_bridge.update_operation(1, 4, f"Building SmartCash YOLOv5 model: {backbone} | Mode: {layer_mode} | Architecture: yolov5")
        else:
            self.logger.info(f"🏗️ Building SmartCash YOLOv5 model: {backbone} | Mode: {layer_mode} | Architecture: yolov5")
        
        # Always use YOLOv5 architecture
        if self.YOLOV5_AVAILABLE and self.model_factory is not None:
            try:
                # Use the model factory to create the model
                model = self.model_factory.create_model(
                    backbone=backbone,
                    num_classes=num_classes,
                    img_size=img_size,
                    pretrained=pretrained,
                    **kwargs
                )
                
                # Apply feature optimization
                if feature_optimization.get('enabled', True):
                    model = self._optimize_model_features(model, feature_optimization)
                
                # Log model info
                model_info = self.get_model_info(model)
                self.logger.info(f"✅ Model built successfully: {model_info.get('parameters', 0):,} parameters")
                
                return model
                
            except Exception as e:
                error_msg = f"Failed to build YOLOv5 model: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        else:
            error_msg = "❌ YOLOv5 integration is required but not available. "
            error_msg += "Please ensure YOLOv5 is properly installed and the model factory is initialized."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
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
            Dictionary containing model information
        """
        info = {
            'parameters': self._count_parameters(model),
            'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') and list(model.parameters()) else 'unknown'
        }
        
        # Add model-specific info if available
        if hasattr(model, 'get_model_info'):
            
            # Try to get YOLOv5-specific model information
            try:
                yolo_model = model.yolov5_model
                
                # Handle YOLOv5 DetectionModel
                if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'model'):
                    # For YOLOv5Wrapper -> DetectionModel -> Sequential
                    detection_model = yolo_model.model
                    if hasattr(detection_model, 'model') and isinstance(detection_model.model, nn.Sequential):
                        model_layers = detection_model.model
                        if len(model_layers) > 0:
                            last_layer = model_layers[-1]
                            if hasattr(last_layer, 'get_layer_info'):
                                info.update(last_layer.get_layer_info())
                elif hasattr(yolo_model, 'model') and isinstance(yolo_model.model, nn.Sequential):
                    # Direct sequential model
                    model_layers = yolo_model.model
                    if len(model_layers) > 0:
                        last_layer = model_layers[-1]
                        if hasattr(last_layer, 'get_layer_info'):
                            info.update(last_layer.get_layer_info())
                
                # Add YOLOv5-specific info if available
                if hasattr(yolo_model, 'names'):
                    info['class_names'] = yolo_model.names
                if hasattr(yolo_model, 'stride'):
                    info['model_stride'] = yolo_model.stride.tolist() if hasattr(yolo_model.stride, 'tolist') else str(yolo_model.stride)
                    
            except Exception as e:
                self.logger.debug(f"Could not extract YOLOv5 model info: {e}")
                
        # Legacy architecture support removed
        
        return info


def create_model(backbone: str = 'cspdarknet', **kwargs) -> nn.Module:
    """
    Convenience function to create SmartCash YOLOv5 model
    
    Args:
        backbone: Backbone type
        **kwargs: Model parameters
        
    Returns:
        YOLOv5 integrated model instance
    """
    builder = ModelBuilder(config={})
    return builder.build(
        backbone=backbone,
        **kwargs
    )


# Export key classes and functions
__all__ = [
    'ModelBuilder',
    'create_model'
]