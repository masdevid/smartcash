"""
File: smartcash/model/architectures/heads/multi_layer_head.py
Description: Multi-layer detection head implementation for banknote detection with 3 detection layers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import HeadError


class MultiLayerHead(nn.Module):
    """
    Multi-layer detection head for banknote detection with 3 distinct layers:
    - Layer 1: Full banknote detection (main object)
    - Layer 2: Nominal-defining features (unique visual cues)
    - Layer 3: Common features (shared among notes)
    """
    
    def __init__(self, in_channels: List[int], layer_specs: Dict[str, Dict] = None,
                 num_anchors: int = 3, img_size: int = 640, use_attention: bool = True,
                 logger: Optional[SmartCashLogger] = None):
        """
        Initialize multi-layer detection head
        
        Args:
            in_channels: List of input channels from backbone [P3, P4, P5]
            layer_specs: Layer specifications for each detection layer
            num_anchors: Number of anchors per scale
            img_size: Input image size
            use_attention: Whether to use channel attention
            logger: Logger instance
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.img_size = img_size
        self.use_attention = use_attention
        
        # Default layer specifications based on README
        if layer_specs is None:
            layer_specs = {
                'layer_1': {
                    'description': 'Full banknote detection (main object)',
                    'classes': ['001', '002', '005', '010', '020', '050', '100'],
                    'num_classes': 7
                },
                'layer_2': {
                    'description': 'Nominal-defining features (unique visual cues)',
                    'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                    'num_classes': 7
                },
                'layer_3': {
                    'description': 'Common features (shared among notes)',
                    'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                    'num_classes': 3
                }
            }
        
        self.layer_specs = layer_specs
        self.layer_names = list(layer_specs.keys())
        
        # Build detection heads for each layer and scale
        self.heads = nn.ModuleDict()
        for layer_name, layer_config in layer_specs.items():
            num_classes = layer_config['num_classes']
            layer_heads = nn.ModuleList()
            
            for ch in in_channels:
                head = self._build_detection_head(ch, num_classes, num_anchors)
                if use_attention:
                    head = self._add_attention_module(head, ch)
                layer_heads.append(head)
            
            self.heads[layer_name] = layer_heads
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(f"✅ Multi-layer head initialized with {len(layer_specs)} layers")
        for layer_name, layer_config in layer_specs.items():
            self.logger.info(f"   • {layer_name}: {layer_config['num_classes']} classes - {layer_config['description']}")
    
    def _build_detection_head(self, in_ch: int, num_classes: int, num_anchors: int) -> nn.Module:
        """Build detection head for a single layer and scale"""
        # YOLOv5-style detection head with improved architecture
        head = nn.Sequential(
            # Feature enhancement layers
            self._conv_block(in_ch, in_ch // 2, 3),
            self._conv_block(in_ch // 2, in_ch // 4, 1),
            self._conv_block(in_ch // 4, in_ch // 2, 3),
            
            # Detection layer
            nn.Conv2d(in_ch // 2, num_anchors * (5 + num_classes), kernel_size=1, bias=True)
        )
        
        return head
    
    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Module:
        """Create convolution block with normalization and activation"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def _add_attention_module(self, head: nn.Module, channels: int) -> nn.Module:
        """Add channel attention to detection head"""
        attention = ChannelAttention(channels // 2)  # Applied to intermediate features
        
        # Wrap head with attention
        class AttentionHead(nn.Module):
            def __init__(self, base_head, attention_module):
                super().__init__()
                self.base_head = base_head
                self.attention = attention_module
            
            def forward(self, x):
                # Apply base head up to last conv layer
                features = self.base_head[:-1](x)
                
                # Apply attention
                attended_features = features * self.attention(features)
                
                # Final detection layer
                return self.base_head[-1](attended_features)
        
        return AttentionHead(head, attention)
    
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Special initialization for detection layer bias
                    if m.out_channels % self.num_anchors == 0:
                        # This is likely a detection layer
                        num_classes = (m.out_channels // self.num_anchors) - 5
                        
                        # Initialize objectness bias
                        obj_idx = 4
                        nn.init.constant_(m.bias[obj_idx::5+num_classes], -math.log((1 - 0.01) / 0.01))
                        
                        # Initialize class bias  
                        if num_classes > 0:
                            cls_start = 5
                            nn.init.constant_(m.bias[cls_start::5+num_classes], -math.log(num_classes - 1))
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass for multi-layer detection
        
        Args:
            features: List of feature maps from backbone [P3, P4, P5]
            
        Returns:
            Dict mapping layer names to prediction lists for each scale
        """
        if len(features) != len(self.in_channels):
            raise HeadError(f"Expected {len(self.in_channels)} feature maps, got {len(features)}")
        
        results = {}
        
        # Determine active layers based on current_phase
        active_layers = self._get_active_layers()
        
        # Log which layers are being computed (only for debugging)
        if len(features) > 0:  # Check if we have features to avoid logging on every call
            current_phase = getattr(self, 'current_phase', 'unknown')
            self.logger.debug(f"🎯 MultiLayerHead Phase {current_phase}: Computing layers {active_layers}")
        
        # Process only active detection layers for performance
        for layer_name in self.layer_names:
            # Skip inactive layers to save computation
            if layer_name not in active_layers:
                if len(features) > 0:  # Only log during actual computation
                    self.logger.debug(f"    ⏭️ Skipping {layer_name} (inactive in phase {getattr(self, 'current_phase', 'unknown')})")
                continue
                
            layer_predictions = []
            layer_heads = self.heads[layer_name]
            num_classes = self.layer_specs[layer_name]['num_classes']
            
            # Process each scale (P3, P4, P5)
            for feat, head in zip(features, layer_heads):
                # Forward through detection head
                pred = head(feat)
                
                # Reshape to YOLO format: [B, anchors, H, W, 5+classes]
                bs, _, h, w = feat.shape
                pred = pred.view(bs, self.num_anchors, 5 + num_classes, h, w)
                pred = pred.permute(0, 1, 3, 4, 2).contiguous()
                
                layer_predictions.append(pred)
            
            results[layer_name] = layer_predictions
        
        return results
    
    def _get_active_layers(self) -> List[str]:
        """
        Get list of active layers based on current_phase.
        
        Returns:
            List of active layer names
        """
        # Check for current_phase attribute on this head
        current_phase = getattr(self, 'current_phase', None)
        
        # If phase not set on head, search through the PyTorch module hierarchy
        if current_phase is None:
            # Try to find current_phase in parent modules by walking up the hierarchy
            current_module = self
            
            # Look for _parent_module or similar attributes that might exist
            parent_attrs = ['_parent_module', '_parent', 'parent', '_owner', 'owner']
            for attr in parent_attrs:
                if hasattr(current_module, attr):
                    parent = getattr(current_module, attr)
                    if hasattr(parent, 'current_phase'):
                        current_phase = getattr(parent, 'current_phase')
                        self.logger.debug(f"Found current_phase={current_phase} from parent {type(parent).__name__}")
                        break
        
        # If still not found, look through global modules for current_phase
        if current_phase is None:
            import gc
            import torch.nn as nn
            
            # Search through all PyTorch modules in memory for current_phase
            for obj in gc.get_objects():
                if (isinstance(obj, nn.Module) and 
                    hasattr(obj, 'current_phase') and 
                    obj.current_phase is not None):
                    current_phase = obj.current_phase
                    break
        
        # Default behavior if no phase info found
        if current_phase is None:
            # Log warning for debugging
            self.logger.debug(f"No current_phase found, defaulting to all layers: {self.layer_names}")
            return self.layer_names  # Use all layers
        
        # Phase-based layer activation
        if current_phase == 1:
            self.logger.debug(f"Phase {current_phase}: activating only layer_1")
            return ['layer_1']  # Phase 1: only layer_1 is active
        elif current_phase == 2:
            self.logger.debug(f"Phase {current_phase}: activating all layers")
            return ['layer_1', 'layer_2', 'layer_3']  # Phase 2: all layers are active
        else:
            # Unknown phase or single-phase mode: use all layers
            self.logger.debug(f"Unknown phase {current_phase}: activating all layers")
            return self.layer_names
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about all detection layers"""
        return {
            'layer_count': len(self.layer_specs),
            'layer_specs': self.layer_specs,
            'total_classes': sum(spec['num_classes'] for spec in self.layer_specs.values()),
            'scales': len(self.in_channels),
            'anchors_per_scale': self.num_anchors,
            'use_attention': self.use_attention,
            'img_size': self.img_size
        }
    
    def get_output_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, List[Tuple]]:
        """Get output shapes for each layer and scale"""
        h, w = input_size
        scales = [8, 16, 32]  # P3, P4, P5 downsampling factors
        
        shapes = {}
        for layer_name, layer_spec in self.layer_specs.items():
            num_classes = layer_spec['num_classes']
            layer_shapes = []
            
            for scale in scales:
                feat_h, feat_w = h // scale, w // scale
                # [batch, anchors, height, width, 5+classes]
                shape = (1, self.num_anchors, feat_h, feat_w, 5 + num_classes)
                layer_shapes.append(shape)
            
            shapes[layer_name] = layer_shapes
        
        return shapes


class ChannelAttention(nn.Module):
    """Channel attention module for feature enhancement"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


def create_multi_layer_head(in_channels: List[int], layer_specs: Dict[str, Dict] = None,
                           **kwargs) -> MultiLayerHead:
    """Factory function to create multi-layer detection head"""
    return MultiLayerHead(in_channels, layer_specs, **kwargs)


def create_banknote_detection_head(in_channels: List[int], **kwargs) -> MultiLayerHead:
    """Create multi-layer head specifically for banknote detection"""
    layer_specs = {
        'layer_1': {
            'description': 'Full banknote detection (main object)',
            'classes': ['001', '002', '005', '010', '020', '050', '100'],
            'num_classes': 7
        },
        'layer_2': {
            'description': 'Nominal-defining features (unique visual cues)', 
            'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
            'num_classes': 7
        },
        'layer_3': {
            'description': 'Common features (shared among notes)',
            'classes': ['l3_sign', 'l3_text', 'l3_thread'],
            'num_classes': 3
        }
    }
    
    return MultiLayerHead(in_channels, layer_specs, **kwargs)