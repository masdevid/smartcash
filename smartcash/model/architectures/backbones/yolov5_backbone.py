"""
YOLOv5-Compatible Backbone Adapters
Adapts SmartCash custom backbones to work with YOLOv5 architecture
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple

# Import YOLOv5 components
yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

try:
    from models.common import Conv, C3, SPPF, Focus
    from utils.general import LOGGER
except ImportError as e:
    print(f"Warning: Could not import YOLOv5 components: {e}")
    # Define fallback components
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, p or k//2, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

from smartcash.common.logger import SmartCashLogger
from smartcash.model.architectures.backbones.cspdarknet import CSPDarknet
from smartcash.model.architectures.backbones.efficientnet import EfficientNetBackbone


class YOLOv5BackboneAdapter(nn.Module):
    """
    Base adapter class that makes SmartCash backbones compatible with YOLOv5
    """
    
    def __init__(self, backbone, feature_indices=None, logger=None):
        """
        Initialize backbone adapter
        
        Args:
            backbone: SmartCash backbone instance
            feature_indices: Indices where to extract features (P3, P4, P5)
            logger: Logger instance
        """
        super().__init__()
        self.backbone = backbone
        self.feature_indices = feature_indices or getattr(backbone, 'feature_indices', [4, 6, 9])
        self.logger = logger or SmartCashLogger(__name__)
        
        # Set YOLOv5 compatibility attributes
        self.save = self.feature_indices
        self.nc = getattr(backbone, 'nc', 1000)  # Default to ImageNet classes
        self.names = [f'class{i}' for i in range(self.nc)]
        
        # Convert backbone layers to YOLOv5 compatible format
        self._setup_yolov5_compatibility()
    
    def _setup_yolov5_compatibility(self):
        """Setup YOLOv5 compatibility attributes for each layer"""
        if hasattr(self.backbone, 'backbone') and hasattr(self.backbone.backbone, '__iter__'):
            # CSPDarknet case
            layers = list(self.backbone.backbone)
        elif hasattr(self.backbone, 'features'):
            # EfficientNet case
            layers = list(self.backbone.features)
        else:
            # Fallback: treat backbone as sequential
            layers = list(self.backbone.children())
        
        # Set YOLOv5 attributes for each layer
        for i, layer in enumerate(layers):
            layer.i = i  # layer index
            layer.f = -1 if i == 0 else [i-1]  # 'from' attribute
            layer.type = layer.__class__.__name__
            layer.np = sum(x.numel() for x in layer.parameters())
        
        self.model = nn.ModuleList(layers)
        self.logger.info(f"🔄 Setup YOLOv5 compatibility for {len(layers)} layers")
    
    def forward(self, x):
        """
        Forward pass with feature extraction at specified indices
        
        Args:
            x: Input tensor or list of tensors
            
        Returns:
            List of feature maps at extraction points
        """
        y = {}  # Dictionary to store layer outputs by index
        features = []
        
        # Handle case where x is a list (from previous layers)
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                x = x[0]  # Unpack single tensor from list
            else:
                # For multiple inputs, use the first one and log a warning
                self.logger.warning("Multiple inputs detected, using the first one")
                x = x[0]
                
        # Initialize with input
        y[-1] = x
        
        for i, m in enumerate(self.model):
            # Get input for current layer
            if m.f == -1:  # Input from previous layer
                xi = y[i-1] if (i-1) in y else x
            elif isinstance(m.f, int):  # Single input
                xi = y.get(m.f, x)
            else:  # Multiple inputs (e.g., concat)
                xi = []
                for j in m.f:
                    if j == -1:
                        xi.append(x)
                    else:
                        xi.append(y.get(j, x))
                if len(xi) == 1:
                    xi = xi[0]
                else:
                    # Ensure we have a single tensor, not a list
                    xi = xi[0] if xi else x
                    
            # Ensure xi is a tensor, not a list
            if isinstance(xi, (list, tuple)):
                xi = xi[0] if xi else x
                
            # Skip if xi is not a tensor
            if not torch.is_tensor(xi):
                self.logger.warning(f"Skipping layer {i} due to invalid input type: {type(xi)}")
                continue
                
            # Forward pass
            try:
                y[i] = m(xi)  # Store output
            except Exception as e:
                self.logger.error(f"Error in layer {i} ({m.__class__.__name__}): {str(e)}")
                raise
            
            # Run layer
            try:
                x = m(xi) if not isinstance(xi, list) else m(*xi)
                y[i] = x  # Save output
                
                # Collect features at specified indices
                if i in self.feature_indices:
                    features.append(x)
                    
            except Exception as e:
                self.logger.error(f"Error in layer {i} ({m.__class__.__name__}): {str(e)}")
                self.logger.error(f"Input shape: {xi.shape if hasattr(xi, 'shape') else 'N/A'}")
                if hasattr(xi, 'dtype'):
                    self.logger.error(f"Input dtype: {xi.dtype}")
                raise
        
        if not features:
            # If no features collected, return the final output
            return [x] if not isinstance(x, (list, tuple)) else list(x)
            
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for feature extraction points"""
        return getattr(self.backbone, 'expected_channels', [256, 512, 1024])


class YOLOv5CSPDarknetAdapter(YOLOv5BackboneAdapter):
    """YOLOv5-compatible adapter for CSPDarknet backbone"""
    
    def __init__(self, model_size='yolov5s', pretrained=True, logger=None, **kwargs):
        """
        Initialize CSPDarknet adapter
        
        Args:
            model_size: YOLOv5 model size ('yolov5s', 'yolov5m', etc.)
            pretrained: Use pretrained weights
            logger: Logger instance (will create one if None)
            **kwargs: Additional arguments for CSPDarknet
        """
        # Ensure we have a logger
        self.logger = logger or SmartCashLogger(__name__)
        
        try:
            # Create CSPDarknet backbone with proper logger
            backbone = CSPDarknet(
                pretrained=pretrained,
                model_size=model_size,
                multi_layer_heads=True,
                logger=self.logger,  # Use the same logger instance
                **kwargs
            )
            
            # Initialize parent with the backbone
            super().__init__(backbone, backbone.feature_indices, self.logger)
            self.model_size = model_size
            
            self.logger.info(f"✅ YOLOv5 CSPDarknet adapter initialized ({model_size})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YOLOv5CSPDarknetAdapter: {str(e)}")
            raise
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dict for YOLOv5 model building"""
        return {
            'backbone_type': 'cspdarknet',
            'model_size': self.model_size,
            'feature_indices': self.feature_indices,
            'output_channels': self.get_output_channels()
        }


class YOLOv5EfficientNetAdapter(YOLOv5BackboneAdapter):
    """
    YOLOv5-compatible adapter for EfficientNet-B4 backbone
    """
    
    def __init__(self, pretrained=True, logger=None, **kwargs):
        """
        Initialize EfficientNet-B4 adapter
        
        Args:
            pretrained: Use pretrained weights
            logger: Logger instance (will create one if None)
            **kwargs: Additional arguments for EfficientNet
        """
        # Ensure we have a logger
        self.logger = logger or SmartCashLogger(__name__)
        
        try:
            # Create EfficientNet backbone with proper logger
            backbone = EfficientNetBackbone(
                pretrained=pretrained,
                multi_layer_heads=True,
                logger=self.logger,  # Use the same logger instance
                **kwargs
            )
            
            # Initialize parent with the backbone
            super().__init__(backbone, backbone.feature_indices, self.logger)
            
            self.logger.info("✅ YOLOv5 EfficientNet-B4 adapter initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YOLOv5EfficientNetAdapter: {str(e)}")
            raise
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dict for YOLOv5 model building"""
        return {
            'backbone_type': 'efficientnet_b4',
            'feature_indices': self.feature_indices,
            'output_channels': self.get_output_channels()
        }


class YOLOv5BackboneFactory:
    """Factory for creating YOLOv5-compatible backbone adapters"""
    
    @staticmethod
    def create_backbone(backbone_type: str, logger=None, **kwargs):
        """
        Create YOLOv5-compatible backbone adapter
        
        Args:
            backbone_type: Type of backbone ('cspdarknet' or 'efficientnet_b4')
            logger: Logger instance (will create one if None)
            **kwargs: Arguments for backbone initialization
            
        Returns:
            YOLOv5BackboneAdapter instance
            
        Raises:
            ValueError: If backbone_type is not supported
            RuntimeError: If backbone creation fails
        """
        # Ensure we have a logger
        logger = logger or SmartCashLogger(__name__)
        backbone_type = backbone_type.lower()
        
        try:
            logger.debug(f"Creating YOLOv5 backbone adapter for type: {backbone_type}")
            
            if backbone_type == 'cspdarknet':
                return YOLOv5CSPDarknetAdapter(logger=logger, **kwargs)
            elif backbone_type == 'efficientnet_b4':
                return YOLOv5EfficientNetAdapter(logger=logger, **kwargs)
            else:
                available = YOLOv5BackboneFactory.get_available_backbones()
                raise ValueError(
                    f"Unsupported backbone type: {backbone_type}. "
                    f"Available types: {', '.join(available)}"
                )
                
        except Exception as e:
            error_msg = f"Failed to create {backbone_type} backbone: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def get_available_backbones():
        """Get list of available backbone types"""
        return ['cspdarknet', 'efficientnet_b4']


def create_yolov5_backbone_from_config(config: Dict[str, Any], logger=None) -> YOLOv5BackboneAdapter:
    """
    Create YOLOv5 backbone from configuration dictionary
    
    Args:
        config: Configuration dictionary with backbone settings
        logger: Optional logger instance (will create one if None)
        
    Returns:
        YOLOv5BackboneAdapter instance
        
    Raises:
        ValueError: If config is invalid or missing required parameters
        RuntimeError: If backbone creation fails
    """
    # Initialize logger
    logger = logger or SmartCashLogger(__name__)
    
    try:
        # Validate config
        if not isinstance(config, dict):
            error_msg = f"Config must be a dictionary, got {type(config).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not config:
            error_msg = "Empty config dictionary provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract backbone type and arguments
        backbone_type = config.get('type', 'cspdarknet')
        backbone_args = {k: v for k, v in config.items() if k != 'type'}
        
        logger.debug(f"Creating YOLOv5 backbone from config: type={backbone_type}, args={backbone_args}")
        
        # Create and return the backbone
        return YOLOv5BackboneFactory.create_backbone(
            backbone_type=backbone_type,
            logger=logger,
            **backbone_args
        )
        
    except Exception as e:
        error_msg = f"Failed to create YOLOv5 backbone from config: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Re-raise with more context if needed
        if not isinstance(e, (ValueError, RuntimeError)):
            raise RuntimeError(error_msg) from e
        raise


# Export functions for easy importing
__all__ = [
    'YOLOv5BackboneAdapter',
    'YOLOv5CSPDarknetAdapter', 
    'YOLOv5EfficientNetAdapter',
    'YOLOv5BackboneFactory',
    'create_yolov5_backbone_from_config'
]