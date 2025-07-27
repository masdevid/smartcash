"""
SmartCash Model Builder with YOLOv5 Integration
Multi-Layer Banknote Detection System
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.utils.progress_bridge import ModelProgressBridge
from smartcash.model.utils.backbone_factory import BackboneFactory
from smartcash.model.core.yolo_head import YOLOHead

# Import YOLOv5 integration components
try:
    from smartcash.model.architectures.yolov5_integration import (
        SmartCashYOLOv5Integration, 
        create_smartcash_yolov5_model,
        create_training_model
    )
    YOLOV5_INTEGRATION_AVAILABLE = True
except ImportError:
    YOLOV5_INTEGRATION_AVAILABLE = False

# Import legacy components from backup file
try:
    from smartcash.model.core.model_builder_legacy import ModelBuilder as LegacyModelBuilder, SmartCashYOLO
except ImportError:
    # If legacy file doesn't exist, create stub classes for compatibility
    class LegacyModelBuilder:
        def __init__(self, *args, **kwargs):
            pass
        def build(self, *args, **kwargs):
            raise NotImplementedError("Legacy model builder not available")
    
    class SmartCashYOLO:
        def __init__(self, *args, **kwargs):
            pass


class ModelBuilder:
    """
    SmartCash model builder with YOLOv5 integration for multi-layer banknote detection
    """
    
    def __init__(self, config: Dict[str, Any], progress_bridge: ModelProgressBridge = None):
        """
        Initialize SmartCash model builder with YOLOv5 integration
        
        Args:
            config: Configuration dictionary
            progress_bridge: Progress tracking bridge
        """
        self.config = config
        self.progress_bridge = progress_bridge
        self.logger = get_logger("model.builder") or get_logger("smartcash")
        
        # Force CPU mode to avoid MPS issues
        self.device = torch.device("cpu")
        self.logger.info("🖥️  Using CPU (MPS disabled due to compatibility issues)")
        
        # Set environment variables to prevent MPS usage
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable high watermark check
        
        # Disable MPS if available to prevent any attempts to use it
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.enabled = False
            torch.backends.mps.is_available = lambda: False
        
        # Initialize legacy builder for fallback
        self.legacy_builder = LegacyModelBuilder(config, progress_bridge) if progress_bridge else None
        
        # Initialize YOLOv5 integration if available
        if YOLOV5_INTEGRATION_AVAILABLE:
            self.yolov5_integration = SmartCashYOLOv5Integration(logger=self.logger)
            self.logger.info("✅ YOLOv5 integration available")
        else:
            self.yolov5_integration = None
            self.logger.warning("⚠️ YOLOv5 integration not available, using legacy mode only")
    
    def build(self, backbone: str = 'efficientnet_b4', detection_layers: List[str] = None,
              layer_mode: str = 'multi', num_classes: int = 7, img_size: int = 640,
              feature_optimization: Dict = None, pretrained: bool = True,
              architecture_type: str = 'auto', **kwargs) -> nn.Module:
        """
        Build model with enhanced architecture selection
        
        Args:
            backbone: Backbone type ('cspdarknet', 'efficientnet_b4')
            detection_layers: Detection layers to use
            layer_mode: 'single', 'multi' for layer mode
            num_classes: Number of classes for primary detection layer
            img_size: Input image size
            feature_optimization: Feature optimization settings
            pretrained: Use pretrained weights
            architecture_type: 'legacy', 'yolov5', 'auto'
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        detection_layers = detection_layers or ['layer_1', 'layer_2', 'layer_3']
        feature_optimization = feature_optimization or {'enabled': True}
        
        self.logger.info(f"🏗️ Building SmartCash YOLOv5 model: {backbone} | Mode: {layer_mode} | Architecture: {architecture_type}")
        
        # Determine architecture type
        if architecture_type == 'auto':
            architecture_type = self._auto_select_architecture(backbone, layer_mode)
        
        if architecture_type == 'yolov5' and self.yolov5_integration:
            return self._build_yolov5_model(
                backbone=backbone,
                detection_layers=detection_layers,
                layer_mode=layer_mode,
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                **kwargs
            )
        else:
            return self._build_legacy_model(
                backbone=backbone,
                detection_layers=detection_layers,
                layer_mode=layer_mode,
                num_classes=num_classes,
                img_size=img_size,
                feature_optimization=feature_optimization,
                pretrained=pretrained,
                **kwargs
            )
    
    def _auto_select_architecture(self, backbone: str, layer_mode: str) -> str:
        """
        Automatically select architecture type based on configuration
        
        Args:
            backbone: Backbone type
            layer_mode: Layer mode
            
        Returns:
            Architecture type ('legacy' or 'yolov5')
        """
        # Prefer YOLOv5 integration if available and using supported backbones
        if (YOLOV5_INTEGRATION_AVAILABLE and 
            backbone in ['cspdarknet', 'efficientnet_b4'] and
            layer_mode == 'multi'):
            return 'yolov5'
        else:
            return 'legacy'
    
    def _optimize_model_features(self, model: nn.Module, optimization_config: Dict) -> nn.Module:
        """
        Apply feature optimization to the model
        
        Args:
            model: The model to optimize
            optimization_config: Optimization configuration
            
        Returns:
            Optimized model
        """
        if not optimization_config.get('enabled', True):
            return model
            
        self.logger.info("🔧 Applying feature optimization...")
        
        # Freeze backbone layers if specified
        if optimization_config.get('freeze_backbone', False):
            for name, param in model.named_parameters():
                if 'model.0.' in name or 'model.1.' in name:  # First few layers are typically the backbone
                    param.requires_grad = False
            self.logger.info("  - Frozen backbone layers")
        
        # Apply other optimizations as needed
        if optimization_config.get('use_amp', False):
            from torch.cuda.amp import autocast
            model.forward = lambda x: model.forward(x.to(next(model.parameters()).device))
            self.logger.info("  - Applied mixed precision training (AMP)")
            
        return model
        
    def _build_yolov5_model(self, backbone: str, detection_layers: List[str],
                           layer_mode: str, num_classes: int, img_size: int,
                           pretrained: bool, feature_optimization: Dict = None, **kwargs) -> nn.Module:
        """
        Build YOLOv5 model with SmartCash integration
        """
        
        if not YOLOV5_INTEGRATION_AVAILABLE or self.yolov5_integration is None:
            raise RuntimeError("YOLOv5 integration is not available")
            
        try:
            if self.progress_bridge:
                self.progress_bridge.update_substep(1, 4, f"🔧 Building YOLOv5 {backbone} model...")
            
            # Prepare model configuration
            model_config = {
                'backbone_type': backbone,
                'num_classes': num_classes,
                'img_size': img_size,
                'pretrained': pretrained,
                'nc': num_classes,  # YOLOv5 uses 'nc' for number of classes
                'ch': 3,  # RGB channels
                'model_size': kwargs.get('model_size', 's'),  # Default to small model
                'anchors': kwargs.get('anchors', None)  # Optional custom anchors
            }
            
            # Filter out None values to avoid passing them to create_training_model
            model_config = {k: v for k, v in model_config.items() if v is not None}
            
            self.logger.info(f"Creating YOLOv5 model with config: {model_config}")
            
            # Create the model using the YOLOv5 integration
            if self.yolov5_integration is not None:
                model = self.yolov5_integration.create_training_compatible_model(
                    backbone_type=backbone,
                    **{k: v for k, v in model_config.items() if k != 'backbone_type'}
                )
            else:
                # Fallback to direct creation if integration is not available
                model = create_training_model(**model_config)
            
            if self.progress_bridge:
                self.progress_bridge.update_substep(2, 4, "🔧 Applying feature optimization...")
            
            # Initialize default optimization config if not provided
            if feature_optimization is None:
                feature_optimization = {'enabled': True}
                
            if feature_optimization.get('enabled', True):
                model = self._optimize_model_features(model, feature_optimization)
                
            if self.progress_bridge:
                self.progress_bridge.update_substep(3, 4, "🔧 Moving model to device...")
                
            # Move model to device if it's a PyTorch model
            if hasattr(model, 'to') and callable(getattr(model, 'to', None)):
                model = model.to(self.device)
            else:
                self.logger.warning("Model does not support .to() method, skipping device move")
            
            if self.progress_bridge:
                self.progress_bridge.update_substep(4, 4, "✅ Model build complete!")
                
            self.logger.info(f"✅ Successfully built YOLOv5 model with {backbone} backbone")
            return model
            
        except Exception as e:
            self.logger.error(f"YOLOv5 model building failed: {e}", exc_info=True)
            raise ValueError(f"Failed to build YOLOv5 model: {str(e)}") from e
    
    def _build_legacy_model(self, backbone: str, detection_layers: List[str],
                           layer_mode: str, num_classes: int, img_size: int,
                           feature_optimization: Dict, pretrained: bool,
                           **kwargs) -> nn.Module:
        """Build model using legacy architecture"""
        
        if self.progress_bridge:
            self.progress_bridge.update_substep(1, 4, f"🔧 Building legacy {backbone} model...")
        
        if self.legacy_builder:
            return self.legacy_builder.build(
                backbone=backbone,
                detection_layers=detection_layers,
                layer_mode=layer_mode,
                num_classes=num_classes,
                img_size=img_size,
                feature_optimization=feature_optimization,
                pretrained=pretrained,
                **kwargs
            )
        else:
            # Fallback implementation
            return self._build_simple_legacy_model(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained
            )
    
    def _build_simple_legacy_model(self, backbone: str, num_classes: int, pretrained: bool):
        """Simple fallback model implementation"""
        
        from smartcash.model.utils.backbone_factory import BackboneFactory
        
        backbone_factory = BackboneFactory()
        
        try:
            backbone_model = backbone_factory.create_backbone(
                backbone,
                pretrained=pretrained,
                feature_optimization=True
            )
            
            # Simple head for compatibility
            from smartcash.model.core.yolo_head import YOLOHead
            
            head = YOLOHead(
                in_channels=backbone_model.get_output_channels(),
                detection_layers=['layer_1'],
                layer_mode='single',
                num_classes=num_classes,
                img_size=640
            )
            
            # Simple neck (identity for now)
            class SimpleNeck(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.channels = channels
                
                def forward(self, x):
                    return x
                
                def get_output_channels(self):
                    return self.channels
            
            neck = SimpleNeck(backbone_model.get_output_channels())
            
            # Assemble model
            model = SmartCashYOLO(
                backbone=backbone_model,
                neck=neck,
                head=head,
                config={
                    'backbone': backbone,
                    'detection_layers': ['layer_1'],
                    'layer_mode': 'single',
                    'num_classes': num_classes,
                    'img_size': 640
                }
            )
            
            self.logger.info(f"✅ Simple legacy model built: {backbone}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Simple model building failed: {str(e)}")
            raise RuntimeError(f"Model building failed: {str(e)}")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in model"""
        return sum(p.numel() for p in model.parameters())
    
    def get_available_architectures(self) -> List[str]:
        """Get list of available architecture types"""
        architectures = ['legacy']
        if YOLOV5_INTEGRATION_AVAILABLE:
            architectures.append('yolov5')
        return architectures
    
    def get_available_backbones(self, architecture_type: str = 'auto') -> List[str]:
        """Get list of available backbones for given architecture"""
        if architecture_type == 'yolov5' and YOLOV5_INTEGRATION_AVAILABLE:
            return ['cspdarknet', 'efficientnet_b4']
        else:
            # Legacy backbones
            return ['efficientnet_b4', 'cspdarknet', 'resnet50']
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'architecture_type': 'unknown'
        }
        
        # Determine architecture type
        if hasattr(model, 'yolov5_model'):
            info['architecture_type'] = 'yolov5_integrated'
            
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
                self.logger.warning(f"Could not extract YOLOv5 model info: {e}")
                
        elif hasattr(model, 'config'):
            info['architecture_type'] = 'legacy'
            if hasattr(model, 'get_model_summary'):
                info.update(model.get_model_summary())
        
        return info


def create_model(backbone: str = 'cspdarknet', architecture_type: str = 'auto',
                 **kwargs) -> nn.Module:
    """
    Convenience function to create SmartCash YOLOv5 model
    
    Args:
        backbone: Backbone type
        architecture_type: Architecture type ('legacy', 'yolov5', 'auto')
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    builder = ModelBuilder(config={})
    return builder.build(
        backbone=backbone,
        architecture_type=architecture_type,
        **kwargs
    )


# Export key classes and functions
__all__ = [
    'ModelBuilder',
    'create_model'
]