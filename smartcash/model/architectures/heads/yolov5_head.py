"""
YOLOv5-Compatible Multi-Layer Detection Head
Integrates SmartCash multi-layer detection with YOLOv5 architecture
"""

import torch
import torch.nn as nn
import math
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional, Tuple

# Import YOLOv5 components
yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

try:
    from models.common import Conv
    from models.yolo import Detect
    from utils.general import LOGGER, check_version
except ImportError as e:
    print(f"Warning: Could not import YOLOv5 components: {e}")
    # Define fallback Detect class
    class Detect(nn.Module):
        stride = None
        dynamic = False
        export = False
        def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
            super().__init__()
            self.nc = nc
            self.no = nc + 5
            self.nl = len(anchors)
            self.na = len(anchors[0]) // 2
            self.grid = [torch.empty(0) for _ in range(self.nl)]
            self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
            self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
            self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
            self.inplace = inplace

from smartcash.common.logger import SmartCashLogger
# Removed unused legacy imports - using YOLOv5 native components instead


class YOLOv5MultiLayerDetect(Detect):
    """
    YOLOv5-compatible multi-layer detection head
    Extends YOLOv5's Detect class to support multiple detection layers
    """
    
    def __init__(self, nc=7, anchors=(), ch=(), inplace=True, **kwargs):
        """
        Initialize multi-layer detection head compatible with YOLOv5
        
        Args:
            nc: Number of classes for primary layer (layer_1)
            anchors: Anchor boxes configuration
            ch: Input channels from neck [P3, P4, P5]
            inplace: Use inplace operations
            **kwargs: Additional keyword arguments including layer_specs
        """
        # Get logger from kwargs or create a new one
        self.logger = kwargs.pop('logger', SmartCashLogger(__name__))
        
        # Extract layer_specs from kwargs if provided
        layer_specs = kwargs.pop('layer_specs', None)
        
        # Default layer specifications for banknote detection
        if layer_specs is None:
            layer_specs = {
                'layer_1': {
                    'nc': nc, 
                    'classes': ['001', '002', '005', '010', '020', '050', '100'],
                    'description': 'Full banknote detection'
                },
                'layer_2': {
                    'nc': 7,
                    'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                    'description': 'Nominal-defining features'
                },
                'layer_3': {
                    'nc': 3,
                    'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                    'description': 'Common features'
                }
            }
        
        self.layer_specs = layer_specs
        self.layer_names = list(layer_specs.keys())
        
        # Ensure ch is a list of integers
        if isinstance(ch, (list, tuple)) and len(ch) > 0 and not isinstance(ch[0], int):
            ch = [x[0] if isinstance(x, (list, tuple)) else x for x in ch]
        
        # Initialize parent Detect class with primary layer settings
        super().__init__(nc=nc, anchors=anchors, ch=ch, inplace=inplace)
        
        # Store the original m (detection layers) from parent
        self.primary_detection = self.m
        
        # Create additional detection heads for other layers
        self.multi_heads = nn.ModuleDict()
        for layer_name, layer_config in layer_specs.items():
            if layer_name != 'layer_1':  # Skip primary layer (handled by parent)
                layer_nc = layer_config['nc']
                layer_no = layer_nc + 5  # bbox + objectness + classes
                
                # Create detection convolutions for this layer
                layer_convs = nn.ModuleList()
                for x in ch:
                    # Ensure x is an integer and create conv layer
                    in_channels = int(x) if not isinstance(x, (list, tuple)) else int(x[0])
                    conv = nn.Conv2d(in_channels, layer_no * self.na, 1)
                    layer_convs.append(conv)
                
                self.multi_heads[layer_name] = layer_convs
        
        self.logger.info(f"✅ Created YOLOv5 multi-layer detection head with {len(self.multi_heads) + 1} layers")
        
        # Initialize weights
        self._initialize_multi_layer_weights()
        
        self.logger.info(f"✅ YOLOv5 Multi-layer head initialized with {len(layer_specs)} layers")
        for layer_name, layer_config in layer_specs.items():
            self.logger.info(f"   • {layer_name}: {layer_config['nc']} classes - {layer_config['description']}")
            
        # Set up model attributes expected by YOLOv5
        self.stride = torch.tensor([8, 16, 32])  # Default stride for P3, P4, P5
        self.export = False  # For ONNX export
        self.training = True  # Start in training mode
    
    def _initialize_multi_layer_weights(self):
        """Initialize weights for multi-layer detection heads"""
        for layer_name, layer_convs in self.multi_heads.items():
            layer_nc = self.layer_specs[layer_name]['nc']
            
            for conv in layer_convs:
                # Initialize bias for better convergence
                b = conv.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / 32) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:5 + layer_nc] += math.log(0.6 / (layer_nc - 0.99999))  # cls
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass through the detection head
        
        Args:
            x: List of feature maps from the neck [P3, P4, P5, ...]
            
        Returns:
            Dictionary of detection outputs for each layer and scale
        """
        # Ensure x is a list of tensors
        if not isinstance(x, (list, tuple)):
            x = [x]
            
        # Process input tensors to ensure they have the correct shape
        processed_x = []
        for xi in x:
            if isinstance(xi, (list, tuple)):
                xi = xi[0] if len(xi) > 0 else None  # Unpack if it's a non-empty list/tuple
                
            if xi is None:
                continue
                
            # Ensure the tensor is 4D [batch, channels, height, width]
            if xi.dim() == 5:
                # If we get a 5D tensor, take the first element of the last dimension
                xi = xi[..., 0]
            elif xi.dim() != 4:
                self.logger.warning(f"Unexpected input tensor dimension: {xi.dim()}, expected 4 or 5")
                continue
                
            # Ensure the tensor is on the correct device
            if hasattr(self, 'device'):
                xi = xi.to(self.device)
                
            processed_x.append(xi)
            
        if not processed_x:
            raise ValueError("No valid input tensors provided")
            
        x = processed_x
        
        # Initialize outputs dictionary
        outputs = {}
        
        # Process primary layer (handled by parent class)
        try:
            # Store the original m (detection layers) and replace with our primary detection
            original_m = self.m
            self.m = self.primary_detection
            
            # Call parent's forward method with all input tensors at once
            # YOLOv5's Detect.forward expects a list of feature maps
            primary_output = super().forward(x)
            
            # Ensure primary_output is a list of tensors
            if torch.is_tensor(primary_output):
                primary_output = [primary_output]
                
            # Process the output to ensure it has the correct shape
            processed_primary = []
            for i, out in enumerate(primary_output):
                # Ensure the output has the expected number of channels
                # The expected number of channels is (num_classes + 5) * num_anchors
                expected_channels = (self.nc + 5) * self.na
                if out.shape[1] != expected_channels:
                    # If the number of channels doesn't match, adjust it
                    adjust_conv = nn.Conv2d(out.shape[1], expected_channels, 1).to(out.device)
                    nn.init.kaiming_normal_(adjust_conv.weight, mode='fan_out', nonlinearity='relu')
                    if adjust_conv.bias is not None:
                        nn.init.constant_(adjust_conv.bias, 0)
                    out = adjust_conv(out)
                
                # Ensure the spatial dimensions match the input
                if out.shape[2:] != x[i].shape[2:]:
                    out = torch.nn.functional.interpolate(
                        out, size=x[i].shape[2:], mode='nearest'
                    )
                
                processed_primary.append(out)
            
            # Store primary output
            outputs['layer_1'] = processed_primary
            
        except Exception as e:
            self.logger.error(f"Error in primary detection head: {e}")
            raise
        finally:
            # Always restore the original m, even if there was an error
            self.m = original_m
        
        # Process additional layers if they exist
        if hasattr(self, 'multi_heads'):
            for layer_name, layer_convs in self.multi_heads.items():
                try:
                    layer_outputs = []
                    for i, conv in enumerate(layer_convs):
                        # Skip if we don't have enough input tensors
                        if i >= len(x):
                            break
                            
                        xi = x[i]
                        
                        # Ensure xi is a 4D tensor
                        if xi.dim() != 4:
                            self.logger.warning(f"Skipping {layer_name} convolution {i}: Expected 4D tensor, got {xi.dim()}D")
                            continue
                        
                        # Get the number of classes for this layer
                        layer_nc = self.layer_specs[layer_name]['nc']
                        expected_channels = (layer_nc + 5) * self.na
                        
                        # Ensure input has the right number of channels for the convolution
                        if xi.shape[1] != conv.in_channels:
                            try:
                                # If channel count doesn't match, use a 1x1 conv to adjust
                                adjust_conv = nn.Conv2d(xi.shape[1], conv.in_channels, 1).to(xi.device)
                                # Initialize weights for the adjustment layer
                                nn.init.kaiming_normal_(adjust_conv.weight, mode='fan_out', nonlinearity='relu')
                                if adjust_conv.bias is not None:
                                    nn.init.constant_(adjust_conv.bias, 0)
                                xi = adjust_conv(xi)
                            except Exception as e:
                                self.logger.error(f"Error adjusting channels for {layer_name} convolution {i}: {e}")
                                continue
                        
                        try:
                            # Apply the layer's convolution
                            layer_output = conv(xi)
                            
                            # Ensure the output has the expected number of channels
                            if layer_output.shape[1] != expected_channels:
                                adjust_conv = nn.Conv2d(layer_output.shape[1], expected_channels, 1).to(layer_output.device)
                                nn.init.kaiming_normal_(adjust_conv.weight, mode='fan_out', nonlinearity='relu')
                                if adjust_conv.bias is not None:
                                    nn.init.constant_(adjust_conv.bias, 0)
                                layer_output = adjust_conv(layer_output)
                            
                            # Ensure the spatial dimensions match the input
                            if layer_output.shape[2:] != xi.shape[2:]:
                                layer_output = torch.nn.functional.interpolate(
                                    layer_output, size=xi.shape[2:], mode='nearest'
                                )
                            
                            layer_outputs.append(layer_output)
                            
                        except Exception as conv_error:
                            self.logger.error(f"Error in {layer_name} convolution {i}: {conv_error}")
                            self.logger.error(f"Input shape: {xi.shape}, Expected input channels: {conv.in_channels}")
                            continue
                    
                    if layer_outputs:  # Only add if we have valid outputs
                        outputs[layer_name] = layer_outputs
                    
                except Exception as e:
                    self.logger.error(f"Error in {layer_name} detection head: {e}")
                    # Continue with other layers even if one fails
        
        return outputs
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about all detection layers"""
        return {
            'layer_count': len(self.layer_specs),
            'layer_specs': self.layer_specs,
            'total_classes': sum(spec['nc'] for spec in self.layer_specs.values()),
            'scales': self.nl,
            'anchors_per_scale': self.na,
            'stride': self.stride.tolist() if self.stride is not None else None
        }


class YOLOv5HeadAdapter:
    """
    Adapter to integrate multi-layer heads with YOLOv5 model building
    """
    
    @staticmethod
    def create_multi_layer_head(ch, nc=7, anchors=(), layer_specs=None, **kwargs):
        """
        Create YOLOv5-compatible multi-layer detection head
        
        Args:
            ch: Input channels from neck
            nc: Number of classes for primary layer
            anchors: Anchor boxes
            layer_specs: Layer specifications
            **kwargs: Additional arguments
            
        Returns:
            YOLOv5MultiLayerDetect instance
        """
        return YOLOv5MultiLayerDetect(
            nc=nc,
            anchors=anchors,
            ch=ch,
            layer_specs=layer_specs,
            **kwargs
        )
    
    @staticmethod
    def create_banknote_head(ch, anchors=(), **kwargs):
        """
        Create multi-layer head specifically for banknote detection
        
        Args:
            ch: Input channels from neck
            anchors: Anchor boxes
            **kwargs: Additional arguments
            
        Returns:
            YOLOv5MultiLayerDetect instance configured for banknotes
        """
        layer_specs = {
            'layer_1': {
                'nc': 7,
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'description': 'Full banknote detection (main object)'
            },
            'layer_2': {
                'nc': 7,
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'description': 'Nominal-defining features (unique visual cues)'
            },
            'layer_3': {
                'nc': 3,
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'description': 'Common features (shared among notes)'
            }
        }
        
        return YOLOv5MultiLayerDetect(
            nc=7,  # Primary layer classes
            anchors=anchors,
            ch=ch,
            layer_specs=layer_specs,
            **kwargs
        )


# Register custom head for YOLOv5 model parsing
def register_yolov5_components():
    """Register custom components with YOLOv5 for model parsing"""
    try:
        import models.yolo as yolo_module
        
        # Add custom detect class to YOLOv5 namespace
        yolo_module.SmartCashMultiDetect = YOLOv5MultiLayerDetect
        
        # Also add to local namespace for eval() in parse_model
        globals()['SmartCashMultiDetect'] = YOLOv5MultiLayerDetect
        
        print("✅ Registered SmartCash components with YOLOv5")
        
    except ImportError:
        print("⚠️ Could not register components - YOLOv5 not available")


# Auto-register components when module is imported
register_yolov5_components()


# Export functions for easy importing
__all__ = [
    'YOLOv5MultiLayerDetect',
    'YOLOv5HeadAdapter',
    'register_yolov5_components'
]