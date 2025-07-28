"""
YOLOv5 Integration Model for SmartCash
Integrates SmartCash custom architectures with YOLOv5 components
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
from typing import Dict, List, Any, Optional

# Import YOLOv5 components
yolov5_path = Path(__file__).parent.parent.parent.parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

try:
    from models.yolo import DetectionModel, parse_model
    from models.common import Conv, C3, SPPF, Concat
    from models.yolo import Detect
    from utils.general import LOGGER
except ImportError as e:
    print(f"Warning: Could not import YOLOv5 components: {e}")
    # Fallback imports for development
    class DetectionModel:
        pass

from smartcash.common.logger import SmartCashLogger
from smartcash.model.architectures.backbones.cspdarknet import CSPDarknet
from smartcash.model.architectures.backbones.efficientnet import EfficientNetB4



class SmartCashYOLOv5(DetectionModel):
    """
    SmartCash YOLOv5 model that integrates custom backbones with YOLOv5 architecture
    """
    
    def __init__(self, cfg="smartcash_yolov5s.yaml", ch=3, nc=None, anchors=None, 
                 backbone_type="cspdarknet", backbone_config=None, logger=None):
        """
        Initialize SmartCash YOLOv5 model
        
        Args:
            cfg: Model configuration (YAML file or dict)
            ch: Input channels
            nc: Number of classes
            anchors: Anchor boxes (list of lists)
            backbone_type: Type of backbone ('cspdarknet' or 'efficientnet_b4')
            backbone_config: Configuration for backbone
            logger: Logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.backbone_type = backbone_type
        self.backbone_config = backbone_config or {}
        
        # Store anchors for later use
        self.custom_anchors = anchors
        
        # If cfg is a string (yaml file), load it
        if isinstance(cfg, str):
            cfg_path = Path(__file__).parent / "configs" / cfg
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f)
            else:
                # Use default config
                cfg = self._get_default_config(backbone_type)
        
        # Initialize parent DetectionModel with anchors=None to prevent rounding issue
        super().__init__(cfg, ch, nc, anchors=None)
        
        # Initialize phase tracking for layer mode control
        self.current_phase = 1  # Default to phase 1 (single layer)
        self.force_single_layer = False  # Flag for explicit single layer mode
        
        # Ensure current_phase is available at all model levels for phase management
        if hasattr(self, 'model') and hasattr(self.model, '__iter__'):
            try:
                # Set current_phase on the detection head (last layer)
                if len(self.model) > 0:
                    last_layer = self.model[-1]
                    if hasattr(last_layer, '__dict__'):
                        last_layer.current_phase = 1
            except:
                pass  # Ignore any errors during initialization
        
        # Apply custom anchors after initialization
        if self.custom_anchors is not None:
            self._apply_custom_anchors()
        
        # Replace backbone with custom implementation
        self._replace_backbone()
        
        self.logger.info(f"✅ SmartCash YOLOv5 initialized with {backbone_type} backbone")
    
    def _get_default_config(self, backbone_type):
        """Get default configuration for the specified backbone"""
        if backbone_type == "cspdarknet":
            return {
                "nc": 7,  # number of classes for banknote detection
                "depth_multiple": 0.33,
                "width_multiple": 0.50,
                "anchors": [
                    [10, 13, 16, 30, 33, 23],  # P3/8
                    [30, 61, 62, 45, 59, 119],  # P4/16
                    [116, 90, 156, 198, 373, 326]  # P5/32
                ],
                "backbone": [
                    # Will be replaced with custom CSPDarknet
                ],
                "head": [
                    [-1, 1, Conv, [512, 1, 1]],
                    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
                    [[-1, 6], 1, Concat, [1]],  # cat backbone P4
                    [-1, 3, C3, [512, False]],  # 13
                    
                    [-1, 1, Conv, [256, 1, 1]],
                    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
                    [[-1, 4], 1, Concat, [1]],  # cat backbone P3
                    [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
                    
                    [-1, 1, Conv, [256, 3, 2]],
                    [[-1, 14], 1, Concat, [1]],  # cat head P4
                    [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
                    
                    [-1, 1, Conv, [512, 3, 2]],
                    [[-1, 10], 1, Concat, [1]],  # cat head P5
                    [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
                    
                    [[17, 20, 23], 1, "SmartCashMultiDetect", [7, "anchors"]],  # Custom multi-layer detect
                ]
            }
        elif backbone_type == "efficientnet_b4":
            # Similar config but adapted for EfficientNet-B4 output channels
            config = self._get_default_config("cspdarknet")
            # Adjust head to match EfficientNet-B4 output channels
            return config
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def _apply_custom_anchors(self):
        """Apply custom anchor boxes to the model"""
        if not self.custom_anchors:
            return
            
        # Convert anchors to proper format if needed
        if isinstance(self.custom_anchors, (list, tuple)):
            if len(self.custom_anchors) > 0 and not isinstance(self.custom_anchors[0], (list, tuple)):
                # Convert flat list to list of lists (3 scales, 3 anchors per scale)
                anchors = [self.custom_anchors[i:i+6] for i in range(0, len(self.custom_anchors), 6)]
            else:
                anchors = self.custom_anchors
            
            # Update model's anchors in the Detect layer(s)
            for m in self.model:
                if hasattr(m, 'anchors') and hasattr(m, 'stride'):
                    m.anchors = torch.tensor(anchors, device=m.anchors.device).float().view(m.nl, -1, 2)
                    m.stride = m.stride.to(m.anchors.device)
                    m.anchor_grid = (m.anchors * m.stride.view(-1, 1, 1)).view(1, -1, 1, 1, 2)
                    break  # Only update the first Detect layer

    def _replace_backbone(self):
        """Replace the standard YOLOv5 backbone with custom implementation"""
        if self.backbone_type == "cspdarknet":
            backbone = CSPDarknet(
                pretrained=self.backbone_config.get('pretrained', True),
                model_size=self.backbone_config.get('model_size', 'yolov5s'),
                multi_layer_heads=True,
                logger=self.logger
            )
        elif self.backbone_type == "efficientnet_b4":
            backbone = EfficientNetB4(
                pretrained=self.backbone_config.get('pretrained', True),
                multi_layer_heads=True,
                logger=self.logger
            )
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
        
        # Extract feature extraction layers (before the head)
        backbone_layers = []
        feature_indices = backbone.feature_indices
        
        # Create YOLOv5-compatible backbone layers
        for i, layer in enumerate(backbone.backbone):
            backbone_layers.append(layer)
            layer.i = i  # Set layer index for YOLOv5 compatibility
            layer.f = -1 if i == 0 else [i-1]  # Set 'from' attribute
            layer.type = layer.__class__.__name__
            layer.np = sum(x.numel() for x in layer.parameters())
        
        # Replace backbone layers in the model
        for i, layer in enumerate(backbone_layers):
            if i < len(self.model):
                self.model[i] = layer
        
        # Update save indices for feature extraction
        self.save = list(feature_indices)
        
        self.logger.info(f"🔄 Replaced backbone with {self.backbone_type}")
        self.logger.info(f"📍 Feature extraction at indices: {feature_indices}")


class SmartCashMultiDetect(nn.Module):
    """
    Multi-layer detection head compatible with YOLOv5 architecture
    Extends YOLOv5's Detect to support multiple detection layers
    """
    
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    
    def __init__(self, nc=7, anchors=(), ch=(), layer_specs=None, inplace=True):
        """
        Initialize multi-layer detection head
        
        Args:
            nc: Number of classes (for layer_1)
            anchors: Anchor boxes
            ch: Input channels from backbone
            layer_specs: Specifications for each detection layer
            inplace: Use inplace operations
        """
        super().__init__()
        
        # Default layer specifications for banknote detection
        if layer_specs is None:
            layer_specs = {
                'layer_1': {'nc': 7, 'description': 'Full banknote detection'},
                'layer_2': {'nc': 7, 'description': 'Nominal-defining features'},
                'layer_3': {'nc': 3, 'description': 'Common features'}
            }
        
        self.layer_specs = layer_specs
        self.layer_names = list(layer_specs.keys())
        self.nl = len(anchors)  # number of detection layers (P3, P4, P5)
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.inplace = inplace
        
        # Create detection heads for each layer and scale
        self.detection_heads = nn.ModuleDict()
        for layer_name, layer_config in layer_specs.items():
            layer_nc = layer_config['nc']
            layer_no = layer_nc + 5  # number of outputs per anchor
            
            # Create detection convolutions for this layer (one for each scale)
            layer_convs = nn.ModuleList([
                nn.Conv2d(x, layer_no * self.na, 1) for x in ch
            ])
            self.detection_heads[layer_name] = layer_convs
    
    def forward(self, x):
        """
        Forward pass through multi-layer detection
        
        Args:
            x: List of feature maps from backbone [P3, P4, P5]
            
        Returns:
            Dict of detection results for each layer, or single layer for training compatibility
        """
        results = {}
        
        # Process each detection layer
        for layer_name, layer_convs in self.detection_heads.items():
            layer_nc = self.layer_specs[layer_name]['nc']
            layer_no = layer_nc + 5
            z = []  # inference output for this layer
            
            for i in range(self.nl):
                # Apply detection convolution
                pred = layer_convs[i](x[i])
                bs, _, ny, nx = pred.shape
                
                # Reshape to YOLOv5 format
                pred = pred.view(bs, self.na, layer_no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                
                if not self.training:  # inference
                    if self.dynamic or self.grid[i].shape[2:4] != pred.shape[2:4]:
                        self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                    
                    # Apply sigmoid and decode predictions
                    xy, wh, conf = pred.sigmoid().split((2, 2, layer_nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                    z.append(y.view(bs, self.na * nx * ny, layer_no))
                else:
                    z.append(pred)
            
            results[layer_name] = z
        
        # For training compatibility, check if we should return single layer or multi-layer
        if self.training:
            # Check if we have a config attribute that determines layer mode
            if hasattr(self, 'force_single_layer') and self.force_single_layer:
                return results['layer_1']
            
            # Check current phase from various possible locations
            current_phase = getattr(self, 'current_phase', None)
            
            # If not found locally, try to get from parent model context
            if current_phase is None:
                # Look for current_phase in the model hierarchy
                try:
                    # Check if we can access the parent model through the call stack
                    import inspect
                    frame = inspect.currentframe()
                    while frame:
                        frame_locals = frame.f_locals
                        if 'self' in frame_locals:
                            parent_obj = frame_locals['self']
                            if hasattr(parent_obj, 'current_phase') and parent_obj != self:
                                current_phase = parent_obj.current_phase
                                break
                        frame = frame.f_back
                except:
                    pass
            
            # Default to phase 1 if still not found
            if current_phase is None:
                current_phase = 1
            
            if current_phase == 1:
                # Phase 1: return only layer_1 for single-layer training
                return results['layer_1']
            else:
                # Phase 2 or multi-layer training: return all layers
                return results
        
        # For inference, return all layers
        return results
    
    def _make_grid(self, nx=20, ny=20, i=0):
        """Generate mesh grid for anchor boxes"""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def create_smartcash_yolov5_model(backbone_type="cspdarknet", model_size="s", 
                                  pretrained=True, num_classes=7, **kwargs):
    """
    Factory function to create SmartCash YOLOv5 model
    
    Args:
        backbone_type: 'cspdarknet' or 'efficientnet_b4'
        model_size: 's', 'm', 'l', 'x' (for CSPDarknet variants)
        pretrained: Use pretrained weights
        num_classes: Number of classes for detection
        **kwargs: Additional arguments
        
    Returns:
        SmartCashYOLOv5 model instance
    """
    backbone_config = {
        'pretrained': pretrained,
        'model_size': f'yolov5{model_size}' if backbone_type == 'cspdarknet' else None
    }
    
    model = SmartCashYOLOv5(
        backbone_type=backbone_type,
        backbone_config=backbone_config,
        nc=num_classes,
        **kwargs
    )
    
    return model