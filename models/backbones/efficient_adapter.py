# File: models/backbones/efficient_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk mengintegrasikan EfficientNet-B4 dengan YOLOv5

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from utils.logger import SmartCashLogger

class EfficientNetAdapter(nn.Module):
    """
    Adapter untuk menghubungkan EfficientNet-B4 dengan YOLOv5.
    Menangani pemetaan feature maps dan transformasi channel dimensions.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Channel dimensions dari EfficientNet-B4
        self.efficient_channels = [56, 160, 448]  # P3, P4, P5
        
        # Channel dimensions yang diharapkan YOLOv5
        self.yolo_channels = [128, 256, 512]  # sesuai CSPDarknet
        
        # Setup adapter layers
        self.adapter_layers = self._build_adapter_layers()
        
    def _build_adapter_layers(self) -> nn.ModuleDict:
        """
        Bangun layer adapter untuk menyesuaikan dimensi channel
        Returns:
            ModuleDict berisi adapter layers
        """
        adapters = OrderedDict()
        
        for i, (in_ch, out_ch) in enumerate(zip(
            self.efficient_channels,
            self.yolo_channels
        )):
            adapters[f'adapter_{i}'] = nn.Sequential(
                # 1x1 conv untuk menyesuaikan channel
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                
                # 3x3 conv untuk meningkatkan receptive field
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
            
        return nn.ModuleDict(adapters)
        
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass adapter
        Args:
            features: List feature maps dari EfficientNet [P3, P4, P5]
        Returns:
            List feature maps yang sudah diadaptasi untuk YOLOv5
        """
        adapted_features = []
        
        for i, feature in enumerate(features):
            adapter = self.adapter_layers[f'adapter_{i}']
            adapted = adapter(feature)
            adapted_features.append(adapted)
            
        return adapted_features

class EfficientNetStage(nn.Module):
    """
    Representasi stage dari EfficientNet untuk ekstraksi feature
    yang lebih granular dan kontrol yang lebih baik.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        expansion_factor: float = 6.0,
        stride: int = 1,
        reduction_ratio: int = 4
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # First block dengan stride
        self.blocks.append(MBConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_factor=expansion_factor,
            stride=stride,
            reduction_ratio=reduction_ratio
        ))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            self.blocks.append(MBConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                expansion_factor=expansion_factor,
                stride=1,
                reduction_ratio=reduction_ratio
            ))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block dengan Squeeze-and-Excitation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: float = 6.0,
        stride: int = 1,
        reduction_ratio: int = 4
    ):
        super().__init__()
        
        self.use_residual = in_channels == out_channels and stride == 1
        expanded_channels = int(in_channels * expansion_factor)
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()
        
        # Depthwise phase
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-excitation
        self.se = SqueezeExcitation(
            expanded_channels,
            reduction_ratio=reduction_ratio
        )
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(p=0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise
        x = self.depthwise(x)
        
        # Squeeze-and-excitation
        x = self.se(x)
        
        # Projection
        x = self.project(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Residual
        if self.use_residual:
            x = x + identity
            
        return x

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block untuk channel attention
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 4
    ):
        super().__init__()
        
        reduced_channels = channels // reduction_ratio
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.squeeze(x)
        attention = self.excitation(attention)
        return x * attention