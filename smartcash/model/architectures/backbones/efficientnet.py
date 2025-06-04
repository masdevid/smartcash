"""
File: smartcash/model/architectures/backbones/efficientnet.py
Deskripsi: Fixed EfficientNet backbone dengan guaranteed 3 feature maps output untuk FPN-PAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import timm

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import BackboneError, UnsupportedBackboneError
from smartcash.model.architectures.backbones.base import BaseBackbone
from smartcash.model.config.model_constants import SUPPORTED_EFFICIENTNET_MODELS, EFFICIENTNET_CHANNELS, YOLO_CHANNELS, DEFAULT_EFFICIENTNET_INDICES

class FeatureAdapter(nn.Module):
    """Adapter untuk memetakan feature maps dari EfficientNet ke format YOLOv5"""
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.attention = ChannelAttention(out_channels) if use_attention else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapt(x)
        return self.attention(x) * x if self.attention is not None else x

class ChannelAttention(nn.Module):
    """Channel Attention untuk memperkuat feature penting"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class EfficientNetBackbone(BaseBackbone):
    """Fixed EfficientNet backbone dengan guaranteed 3 feature maps untuk FPN-PAN"""
    
    def __init__(self, model_name: str = 'efficientnet_b4', pretrained: bool = True, 
                 feature_indices: Optional[List[int]] = None, out_channels: Optional[List[int]] = None,
                 use_attention: bool = True, testing_mode: bool = False, logger: Optional[SmartCashLogger] = None):
        super().__init__(logger=logger)
        
        self.model_name = model_name
        self.testing_mode = testing_mode
        self.feature_indices = feature_indices or DEFAULT_EFFICIENTNET_INDICES
        self.use_attention = use_attention
        
        # Validate model dengan one-liner
        model_name in SUPPORTED_EFFICIENTNET_MODELS or self._raise_error(f"❌ Model {model_name} tidak didukung. Model yang didukung: {', '.join(SUPPORTED_EFFICIENTNET_MODELS)}")
        
        # Setup output channels - ALWAYS ensure 3 feature maps
        self.out_channels = out_channels or YOLO_CHANNELS
        len(self.out_channels) == 3 or self._raise_error(f"❌ Output channels harus 3 untuk FPN-PAN, ditemukan {len(self.out_channels)}")
        
        # Initialize model berdasarkan mode
        if testing_mode:
            self._setup_dummy_model()
        else:
            self._setup_real_model(pretrained)
        
        self.logger.success(f"✅ EfficientNet {model_name} initialized: {len(self.out_channels)} feature maps -> {self.out_channels}")
    
    def _setup_real_model(self, pretrained: bool):
        """Setup real EfficientNet model dengan guaranteed feature maps"""
        try:
            # Create base model dengan timm
            self.model = timm.create_model(
                self.model_name, 
                pretrained=pretrained, 
                features_only=True, 
                out_indices=self.feature_indices
            )
            
            # Get actual channels dari model
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                features = self.model(dummy_input)
                self.actual_channels = [f.shape[1] for f in features]
            
            # Validate feature count - CRITICAL untuk FPN-PAN
            len(features) == 3 or self._raise_error(f"❌ EfficientNet harus menghasilkan 3 feature maps, ditemukan {len(features)}")
            
            # Create adapters untuk convert ke YOLO channels
            self.adapters = nn.ModuleList([
                FeatureAdapter(in_ch, out_ch, self.use_attention) 
                for in_ch, out_ch in zip(self.actual_channels, self.out_channels)
            ])
            
            self.logger.info(f"🔧 Feature adapters: {self.actual_channels} -> {self.out_channels}")
            
        except Exception as e:
            self._raise_error(f"❌ Error setup EfficientNet model: {str(e)}")
    
    def _setup_dummy_model(self):
        """Setup dummy model untuk testing dengan guaranteed 3 outputs"""
        self.logger.info(f"🧪 Creating dummy {self.model_name} dengan {len(self.out_channels)} feature maps")
        
        # Create dummy layers yang menghasilkan tepat 3 feature maps
        self.dummy_layers = nn.ModuleList()
        in_channels = 3
        
        for i, out_ch in enumerate(self.out_channels):
            # Calculate proper stride untuk feature map sizes (1/8, 1/16, 1/32)
            stride = 2 ** (i + 3)  # 8, 16, 32
            
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((80 // (2**i), 80 // (2**i)))  # Proper feature map sizes
            )
            self.dummy_layers.append(layer)
            in_channels = out_ch
        
        self.logger.info(f"✅ Dummy model: {len(self.dummy_layers)} layers -> {self.out_channels} channels")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass dengan guaranteed 3 feature maps output"""
        if self.testing_mode:
            return self._forward_dummy(x)
        
        try:
            # Extract features dari EfficientNet
            features = self.model(x)
            
            # Validate feature count - CRITICAL CHECK
            len(features) == 3 or self._raise_error(f"❌ Expected 3 features, got {len(features)} - FPN-PAN requires exactly 3 feature maps")
            
            # Apply adapters untuk convert ke YOLO format
            adapted_features = [adapter(feat) for feat, adapter in zip(features, self.adapters)]
            
            # Final validation
            len(adapted_features) == 3 or self._raise_error(f"❌ Adapter output count mismatch: {len(adapted_features)}")
            
            # Validate output channels
            actual_out_channels = [f.shape[1] for f in adapted_features]
            actual_out_channels == self.out_channels or self.logger.warning(f"⚠️ Channel mismatch: expected {self.out_channels}, got {actual_out_channels}")
            
            self.logger.debug(f"🔍 Feature shapes: {[f.shape for f in adapted_features]}")
            return adapted_features
            
        except Exception as e:
            self._raise_error(f"❌ Forward pass error: {str(e)}")
    
    def _forward_dummy(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Dummy forward dengan guaranteed 3 feature maps"""
        features = []
        current = x
        
        for i, layer in enumerate(self.dummy_layers):
            # Progressive downsampling untuk simulate real feature pyramid
            if i > 0:
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
            
            # Apply layer transformation
            feat = layer(current)
            features.append(feat)
            
            self.logger.debug(f"🧪 Dummy feature {i}: {feat.shape}")
        
        # Final validation untuk dummy
        len(features) == 3 or self._raise_error(f"❌ Dummy model harus menghasilkan 3 features, got {len(features)}")
        
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channels dengan validation"""
        return self.out_channels
    
    def get_output_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> List[Tuple[int, int]]:
        """Get output shapes untuk feature maps"""
        width, height = input_size
        return [(height // 8, width // 8), (height // 16, width // 16), (height // 32, width // 32)]  # P3, P4, P5
    
    def get_info(self) -> Dict:
        """Get backbone info dengan feature validation"""
        return {
            'type': 'EfficientNet',
            'variant': self.model_name,
            'out_channels': self.out_channels,
            'feature_indices': self.feature_indices,
            'actual_channels': getattr(self, 'actual_channels', self.out_channels),
            'pretrained': not self.testing_mode,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'feature_count': len(self.out_channels),
            'fpn_compatible': len(self.out_channels) == 3
        }
    
    def _raise_error(self, message: str):
        """Raise BackboneError dengan logging"""
        self.logger.error(message)
        raise BackboneError(message)

# One-liner utilities untuk backbone validation
validate_feature_count = lambda features: len(features) == 3 or (_ for _ in ()).throw(ValueError(f"FPN-PAN requires 3 features, got {len(features)}"))
create_feature_adapter = lambda in_ch, out_ch, use_attention: FeatureAdapter(in_ch, out_ch, use_attention)
get_dummy_feature_shape = lambda input_shape, level: (input_shape[0] // (8 * (2 ** level)), input_shape[1] // (8 * (2 ** level)))