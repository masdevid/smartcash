# File: models/necks/fpn_pan.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi Feature Pyramid Network dan Path Aggregation Network 
# untuk memproses fitur dari EfficientNet-B4 backbone

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from smartcash.utils.logger import SmartCashLogger

class FeatureProcessingNeck(nn.Module):
    """
    Modul untuk memproses dan menggabungkan fitur dari berbagai skala
    menggunakan kombinasi FPN dan PAN
    """
    
    def __init__(
        self,
        in_channels: List[int],  # Channel dari EfficientNet stages [56, 160, 448]
        out_channels: List[int],  # Channel output untuk YOLOv5 [128, 256, 512]
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        # Path Aggregation Network
        self.pan = PathAggregationNetwork(
            in_channels=out_channels,
            out_channels=out_channels
        )
        
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network untuk mengkombinasikan fitur dari berbagai level
    dengan koneksi top-down
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int]
    ):
        super().__init__()
        
        # Lateral connections (1x1 conv untuk menyesuaikan channel)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        
        # Top-down connections (3x3 conv setelah upsampling)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
            for out_ch in out_channels
        ])
        
        # Upsampling untuk top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass FPN
        Args:
            features: List fitur dari backbone [P3, P4, P5]
        Returns:
            List fitur yang telah diproses FPN
        """
        # Lateral connections
        laterals = [
            conv(feature)
            for feature, conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway
        fpn_features = [laterals[-1]]  # Start dengan level tertinggi
        for i in range(len(laterals)-2, -1, -1):
            # Upsample fitur level atas
            top_down = self.upsample(fpn_features[0])
            # Tambahkan dengan lateral connection
            fpn_feature = laterals[i] + top_down
            # Aplikasikan 3x3 conv
            fpn_feature = self.fpn_convs[i](fpn_feature)
            fpn_features.insert(0, fpn_feature)
            
        return fpn_features

class PathAggregationNetwork(nn.Module):
    """
    Path Aggregation Network untuk bottom-up path augmentation
    yang memperkuat propagasi fitur lokal
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int]
    ):
        super().__init__()
        
        # Bottom-up convolutions
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch*2, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
            for in_ch, out_ch in zip(in_channels[:-1], out_channels[1:])
        ])
        
        # Downsampling untuk bottom-up pathway
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass PAN
        Args:
            fpn_features: List fitur dari FPN [P3, P4, P5]
        Returns:
            List fitur yang telah diproses PAN
        """
        pan_features = [fpn_features[0]]  # Start dengan level terendah
        
        # Bottom-up pathway
        for i in range(len(fpn_features)-1):
            # Downsample fitur level bawah
            bottom_up = self.downsample(pan_features[-1])
            # Concat dengan fpn feature
            combined = torch.cat([bottom_up, fpn_features[i+1]], dim=1)
            # Aplikasikan conv
            pan_feature = self.bu_convs[i](combined)
            pan_features.append(pan_feature)
            
        return pan_features