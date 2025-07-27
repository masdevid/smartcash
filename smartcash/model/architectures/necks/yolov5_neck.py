"""
YOLOv5-Compatible Neck Implementation
Adapts SmartCash FPN-PAN neck to work with YOLOv5 architecture
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

# Import YOLOv5 components
yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

try:
    from models.common import Conv, C3, Concat
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
    
    class C3(nn.Module):
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__()
            self.cv1 = Conv(c1, c2, 1, 1)
        def forward(self, x):
            return self.cv1(x)
    
    class Concat(nn.Module):
        def __init__(self, dimension=1):
            super().__init__()
            self.d = dimension
        def forward(self, x):
            return torch.cat(x, self.d)

from smartcash.common.logger import SmartCashLogger


class YOLOv5FPNPANNeck(nn.Module):
    """
    YOLOv5-compatible FPN-PAN neck implementation
    Based on YOLOv5's head architecture but using SmartCash design principles
    """
    
    def __init__(self, in_channels, out_channels=None, logger=None):
        """
        Initialize YOLOv5-compatible FPN-PAN neck
        
        Args:
            in_channels: Input channels from backbone [P3, P4, P5]
            out_channels: Output channels for detection heads [P3, P4, P5]
            logger: Logger instance
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.in_channels = in_channels
        self.out_channels = out_channels or [256, 512, 1024]
        
        # Ensure we have 3 feature levels
        if len(in_channels) != 3:
            raise ValueError(f"Expected 3 input channels, got {len(in_channels)}")
        
        c3, c4, c5 = in_channels  # P3, P4, P5 channels from backbone
        o3, o4, o5 = self.out_channels  # Output channels
        
        # FPN (top-down pathway)
        # P5 -> P4
        self.cv1 = Conv(c5, o4, 1, 1)  # reduce P5 channels
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.concat1 = Concat(1)  # cat backbone P4
        self.cv2 = C3(c4 + o4, o4, shortcut=False)  # C3 after concat
        
        # P4 -> P3
        self.cv3 = Conv(o4, o3, 1, 1)  # reduce P4 channels  
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.concat2 = Concat(1)  # cat backbone P3
        self.cv4 = C3(c3 + o3, o3, shortcut=False)  # C3 after concat (P3/8-small)
        
        # PAN (bottom-up pathway)
        # P3 -> P4
        self.cv5 = Conv(o3, o3, 3, 2)  # downsample P3
        self.concat3 = Concat(1)  # cat head P4
        self.cv6 = C3(o4 + o3, o4, shortcut=False)  # C3 after concat (P4/16-medium)
        
        # P4 -> P5
        self.cv7 = Conv(o4, o4, 3, 2)  # downsample P4
        self.concat4 = Concat(1)  # cat head P5
        self.cv8 = C3(o5 + o4, o5, shortcut=False)  # C3 after concat (P5/32-large)
        
        # Set YOLOv5 compatibility attributes
        self._setup_yolov5_compatibility()
        
        self.logger.info(f"✅ YOLOv5 FPN-PAN neck initialized: {in_channels} -> {self.out_channels}")
    
    def _setup_yolov5_compatibility(self):
        """Setup YOLOv5 compatibility attributes"""
        # Create a list of all modules for YOLOv5 compatibility
        self.model = nn.ModuleList([
            self.cv1,      # 0
            self.upsample1, # 1
            self.concat1,   # 2
            self.cv2,       # 3
            self.cv3,       # 4
            self.upsample2, # 5
            self.concat2,   # 6
            self.cv4,       # 7 (P3/8-small)
            self.cv5,       # 8
            self.concat3,   # 9
            self.cv6,       # 10 (P4/16-medium)
            self.cv7,       # 11
            self.concat4,   # 12
            self.cv8,       # 13 (P5/32-large)
        ])
        
        # Set YOLOv5 attributes for each layer
        layer_configs = [
            ([-1], 'Conv'),           # 0: cv1
            ([-1], 'Upsample'),      # 1: upsample1
            ([[-1, 6]], 'Concat'),   # 2: concat1 (cat backbone P4)
            ([-1], 'C3'),            # 3: cv2
            ([-1], 'Conv'),          # 4: cv3
            ([-1], 'Upsample'),      # 5: upsample2
            ([[-1, 4]], 'Concat'),   # 6: concat2 (cat backbone P3)
            ([-1], 'C3'),            # 7: cv4 (P3/8-small)
            ([-1], 'Conv'),          # 8: cv5
            ([[-1, 14]], 'Concat'),  # 9: concat3 (cat head P4)
            ([-1], 'C3'),            # 10: cv6 (P4/16-medium)
            ([-1], 'Conv'),          # 11: cv7
            ([[-1, 10]], 'Concat'),  # 12: concat4 (cat head P5)
            ([-1], 'C3'),            # 13: cv8 (P5/32-large)
        ]
        
        for i, (layer, (f, type_name)) in enumerate(zip(self.model, layer_configs)):
            layer.i = i  # layer index
            layer.f = f  # 'from' attribute
            layer.type = type_name
            layer.np = sum(x.numel() for x in layer.parameters())
        
        # Save indices for output features
        self.save = [7, 10, 13]  # P3, P4, P5 output indices
    
    def forward(self, x):
        """
        Forward pass through FPN-PAN neck
        
        Args:
            x: List of feature maps from backbone [P3, P4, P5]
            
        Returns:
            List of processed feature maps [P3_out, P4_out, P5_out]
        """
        if len(x) != 3:
            raise ValueError(f"Expected 3 input features, got {len(x)}")
        
        p3, p4, p5 = x
        
        # FPN (top-down pathway)
        # P5 -> P4
        p5_reduced = self.cv1(p5)  # reduce channels
        p5_up = self.upsample1(p5_reduced)  # upsample
        p4_concat = self.concat1([p5_up, p4])  # concat with backbone P4
        p4_out = self.cv2(p4_concat)  # process
        
        # P4 -> P3  
        p4_reduced = self.cv3(p4_out)  # reduce channels
        p4_up = self.upsample2(p4_reduced)  # upsample
        p3_concat = self.concat2([p4_up, p3])  # concat with backbone P3
        p3_out = self.cv4(p3_concat)  # P3/8-small output
        
        # PAN (bottom-up pathway)
        # P3 -> P4
        p3_down = self.cv5(p3_out)  # downsample P3
        p4_concat2 = self.concat3([p3_down, p4_out])  # concat with head P4
        p4_final = self.cv6(p4_concat2)  # P4/16-medium output
        
        # P4 -> P5
        p4_down = self.cv7(p4_final)  # downsample P4
        p5_concat = self.concat4([p4_down, p5_reduced])  # concat with head P5
        p5_final = self.cv8(p5_concat)  # P5/32-large output
        
        return [p3_out, p4_final, p5_final]
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for each scale"""
        return self.out_channels


class YOLOv5NeckAdapter:
    """
    Adapter for creating YOLOv5-compatible necks
    """
    
    @staticmethod
    def create_fpn_pan_neck(in_channels, out_channels=None, **kwargs):
        """
        Create YOLOv5-compatible FPN-PAN neck
        
        Args:
            in_channels: Input channels from backbone
            out_channels: Output channels for detection heads
            **kwargs: Additional arguments
            
        Returns:
            YOLOv5FPNPANNeck instance
        """
        return YOLOv5FPNPANNeck(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
    
    @staticmethod
    def create_neck_from_config(config: Dict[str, Any]):
        """
        Create neck from configuration dictionary
        
        Args:
            config: Configuration dictionary
            
        Returns:
            YOLOv5FPNPANNeck instance
        """
        neck_type = config.get('neck_type', 'fpn_pan')
        
        if neck_type.lower() == 'fpn_pan':
            return YOLOv5NeckAdapter.create_fpn_pan_neck(
                in_channels=config.get('in_channels'),
                out_channels=config.get('out_channels'),
                logger=config.get('logger')
            )
        else:
            raise ValueError(f"Unsupported neck type: {neck_type}")


def create_yolov5_neck_layers(in_channels, out_channels=None):
    """
    Create YOLOv5 neck layers for model building
    
    Args:
        in_channels: Input channels from backbone [P3, P4, P5]
        out_channels: Output channels [P3, P4, P5]
        
    Returns:
        List of layer specifications for YOLOv5 model building
    """
    if out_channels is None:
        out_channels = [256, 512, 1024]
    
    c3, c4, c5 = in_channels
    o3, o4, o5 = out_channels
    
    # YOLOv5 head specification
    head_layers = [
        [-1, 1, Conv, [o4, 1, 1]],                    # 0: reduce P5 channels
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],   # 1: upsample
        [[-1, 6], 1, Concat, [1]],                    # 2: cat backbone P4  
        [-1, 3, C3, [o4, False]],                     # 3: C3
        
        [-1, 1, Conv, [o3, 1, 1]],                    # 4: reduce P4 channels
        [-1, 1, nn.Upsample, [None, 2, "nearest"]],   # 5: upsample
        [[-1, 4], 1, Concat, [1]],                    # 6: cat backbone P3
        [-1, 3, C3, [o3, False]],                     # 7: P3/8-small
        
        [-1, 1, Conv, [o3, 3, 2]],                    # 8: downsample
        [[-1, 14], 1, Concat, [1]],                   # 9: cat head P4
        [-1, 3, C3, [o4, False]],                     # 10: P4/16-medium
        
        [-1, 1, Conv, [o4, 3, 2]],                    # 11: downsample
        [[-1, 10], 1, Concat, [1]],                   # 12: cat head P5
        [-1, 3, C3, [o5, False]],                     # 13: P5/32-large
    ]
    
    return head_layers


# Export functions for easy importing
__all__ = [
    'YOLOv5FPNPANNeck',
    'YOLOv5NeckAdapter', 
    'create_yolov5_neck_layers'
]