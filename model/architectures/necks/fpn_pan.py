"""
File: smartcash/model/architectures/necks/fpn_pan.py
Deskripsi: Fixed FPN-PAN dengan robust input validation dan guaranteed 3 feature maps
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import NeckError
from smartcash.model.config.model_constants import YOLO_CHANNELS

class ConvBlock(nn.Module):
    """Convolution block dengan BatchNorm dan aktivasi"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block untuk mempertahankan informasi"""
    
    def __init__(self, channels: int):
        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvBlock(channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBlock(mid_channels, channels, kernel_size=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))

class FPN_PAN(nn.Module):
    """Fixed FPN-PAN dengan robust input validation dan guaranteed output"""
    
    def __init__(self, in_channels: List[int], out_channels: Optional[List[int]] = None, logger: Optional[SmartCashLogger] = None):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.in_channels = in_channels or []
        self.out_channels = out_channels or YOLO_CHANNELS
        
        # Critical validation dengan detailed error messages
        if len(self.in_channels) == 0:
            self._raise_error("❌ Input channels kosong! Backbone tidak menghasilkan feature maps.")
        
        if len(self.in_channels) != 3:
            self._raise_error(f"❌ FPN-PAN membutuhkan 3 input feature maps, tetapi {len(self.in_channels)} diberikan. Channels: {self.in_channels}")
        
        if len(self.out_channels) != 3:
            self._raise_error(f"❌ FPN-PAN membutuhkan 3 output feature maps, tetapi {len(self.out_channels)} diberikan. Channels: {self.out_channels}")
        
        # Validate channel values
        if any(ch <= 0 for ch in self.in_channels):
            self._raise_error(f"❌ Input channels harus > 0, ditemukan: {self.in_channels}")
        
        if any(ch <= 0 for ch in self.out_channels):
            self._raise_error(f"❌ Output channels harus > 0, ditemukan: {self.out_channels}")
        
        # Initialize FPN dan PAN components
        try:
            self.fpn = FeaturePyramidNetwork(in_channels=self.in_channels, out_channels=self.out_channels)
            self.pan = PathAggregationNetwork(in_channels=self.out_channels, out_channels=self.out_channels)
            
            self.logger.success(f"✅ FPN-PAN initialized: {self.in_channels} -> {self.out_channels}")
            
        except Exception as e:
            self._raise_error(f"❌ Error initializing FPN-PAN components: {str(e)}")
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass dengan robust validation"""
        # Input validation dengan detailed checks
        if not features:
            self._raise_error("❌ Input features kosong! Backbone tidak menghasilkan output.")
        
        if len(features) != 3:
            self._raise_error(f"❌ Expected 3 feature maps, got {len(features)}. Feature shapes: {[f.shape if isinstance(f, torch.Tensor) else type(f) for f in features]}")
        
        # Validate feature tensor properties
        for i, feat in enumerate(features):
            if not isinstance(feat, torch.Tensor):
                self._raise_error(f"❌ Feature {i} bukan torch.Tensor: {type(feat)}")
            
            if feat.dim() != 4:
                self._raise_error(f"❌ Feature {i} harus 4D (B,C,H,W), got {feat.dim()}D: {feat.shape}")
            
            if feat.shape[1] != self.in_channels[i]:
                self._raise_error(f"❌ Feature {i} channel mismatch: expected {self.in_channels[i]}, got {feat.shape[1]}")
        
        try:
            # Log input info untuk debugging
            self.logger.debug(f"🔍 FPN-PAN input shapes: {[f.shape for f in features]}")
            
            # Process melalui FPN dan PAN
            fpn_features = self.fpn(features)
            pan_features = self.pan(fpn_features)
            
            # Output validation
            if len(pan_features) != 3:
                self._raise_error(f"❌ PAN output count mismatch: expected 3, got {len(pan_features)}")
            
            # Validate output channels
            actual_out_channels = [f.shape[1] for f in pan_features]
            if actual_out_channels != self.out_channels:
                self.logger.warning(f"⚠️ Output channel mismatch: expected {self.out_channels}, got {actual_out_channels}")
            
            self.logger.debug(f"🔍 FPN-PAN output shapes: {[f.shape for f in pan_features]}")
            return pan_features
            
        except Exception as e:
            self._raise_error(f"❌ FPN-PAN forward pass error: {str(e)}")
    
    def get_output_channels(self) -> List[int]:
        """Get output channels dengan validation"""
        return self.out_channels
    
    def _raise_error(self, message: str):
        """Raise NeckError dengan logging"""
        self.logger.error(message)
        raise NeckError(message)

class FeaturePyramidNetwork(nn.Module):
    """Fixed FPN dengan robust channel handling"""
    
    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()
        
        # Validate inputs
        if len(in_channels) != len(out_channels):
            raise ValueError(f"Channel length mismatch: in={len(in_channels)}, out={len(out_channels)}")
        
        # Lateral connections dengan proper channel mapping
        self.lateral_convs = nn.ModuleList([
            ConvBlock(in_ch, out_ch, kernel_size=1) 
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        
        # FPN blocks untuk feature enhancement
        self.fpn_blocks = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(out_ch) for _ in range(2)]) 
            for out_ch in out_channels
        ])
        
        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Channel adapters untuk size mismatch
        self.channel_adapters = nn.ModuleDict()
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """FPN forward dengan adaptive channel handling"""
        # Lateral connections
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        
        # Top-down pathway starting dari level tertinggi (P5)
        fpn_features = [self.fpn_blocks[-1](laterals[-1])]
        
        # Process dari atas ke bawah
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample feature dari level atas
            top_down = self.upsample(fpn_features[0])
            
            # Adaptive channel matching
            if top_down.shape[1] != laterals[i].shape[1]:
                adapter_key = f"adapter_{top_down.shape[1]}_to_{laterals[i].shape[1]}"
                if adapter_key not in self.channel_adapters:
                    self.channel_adapters[adapter_key] = nn.Conv2d(
                        top_down.shape[1], laterals[i].shape[1], 
                        kernel_size=1, bias=False
                    ).to(top_down.device)
                top_down = self.channel_adapters[adapter_key](top_down)
            
            # Spatial size matching dengan adaptive pooling
            if top_down.shape[2:] != laterals[i].shape[2:]:
                top_down = nn.functional.adaptive_avg_pool2d(top_down, laterals[i].shape[2:])
            
            # Combine lateral dan top-down
            fpn_feature = self.fpn_blocks[i](laterals[i] + top_down)
            fpn_features.insert(0, fpn_feature)
        
        return fpn_features

class PathAggregationNetwork(nn.Module):
    """Fixed PAN dengan adaptive channel handling"""
    
    def __init__(self, in_channels: List[int], out_channels: List[int], num_repeats: int = 2):
        super().__init__()
        
        # Validate inputs
        if len(in_channels) != len(out_channels):
            raise ValueError(f"Channel length mismatch: in={len(in_channels)}, out={len(out_channels)}")
        
        # Downsampling layers untuk bottom-up pathway
        self.downsamples = nn.ModuleList([
            ConvBlock(in_ch, in_ch, kernel_size=3, stride=2) 
            for in_ch in in_channels[:-1]
        ])
        
        # Bottom-up convolutions dengan adaptive channel handling
        self.bu_convs = nn.ModuleList([
            ConvBlock(in_ch * 2, out_ch, kernel_size=1) 
            for in_ch, out_ch in zip(in_channels[1:], out_channels[1:])
        ])
        
        # PAN enhancement blocks
        self.pan_blocks = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(out_ch) for _ in range(num_repeats)]) 
            for out_ch in out_channels
        ])
        
        # Channel adapters untuk dynamic channel matching
        self.channel_adapters = nn.ModuleDict()
    
    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """PAN forward dengan adaptive processing"""
        # Start dengan level terendah (P3)
        pan_features = [self.pan_blocks[0](fpn_features[0])]
        
        # Bottom-up pathway
        for i in range(len(fpn_features) - 1):
            # Downsample dari level bawah
            bottom_up = self.downsamples[i](pan_features[-1])
            
            # Combine dengan FPN feature
            expected_channels = self.bu_convs[i].conv.in_channels
            actual_channels = bottom_up.shape[1] + fpn_features[i + 1].shape[1]
            
            # Adaptive channel matching
            if actual_channels != expected_channels:
                adapter_key = f"bu_adapter_{actual_channels}_to_{expected_channels}"
                if adapter_key not in self.channel_adapters:
                    self.channel_adapters[adapter_key] = nn.Conv2d(
                        actual_channels, expected_channels, 
                        kernel_size=1, bias=False
                    ).to(bottom_up.device)
                
                combined = torch.cat([bottom_up, fpn_features[i + 1]], dim=1)
                combined = self.channel_adapters[adapter_key](combined)
            else:
                combined = torch.cat([bottom_up, fpn_features[i + 1]], dim=1)
            
            # Apply bottom-up conv dan enhancement
            pan_feature = self.pan_blocks[i + 1](self.bu_convs[i](combined))
            pan_features.append(pan_feature)
        
        return pan_features

# One-liner utilities untuk FPN-PAN validation
validate_feature_list = lambda features: len(features) == 3 or (_ for _ in ()).throw(NeckError(f"Expected 3 features, got {len(features)}"))
validate_channels = lambda channels: all(ch > 0 for ch in channels) or (_ for _ in ()).throw(NeckError(f"Invalid channels: {channels}"))
create_channel_adapter = lambda in_ch, out_ch: nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
safe_upsample = lambda x, target_size: nn.functional.adaptive_avg_pool2d(x, target_size) if x.shape[2:] != target_size else x