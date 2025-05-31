"""
File: smartcash/model/architectures/necks/fpn_pan.py
Deskripsi: Feature Processing Neck implementation for YOLOv5
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import NeckError
from smartcash.model.config.model_constants import YOLO_CHANNELS

class ConvBlock(nn.Module):
    """Convolution block dengan BatchNorm dan aktivasi."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None):
        """Inisialisasi Conv Block."""
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2  # Same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block untuk mempertahankan informasi."""
    
    def __init__(self, channels: int):
        """Inisialisasi Residual Block."""
        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvBlock(channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBlock(mid_channels, channels, kernel_size=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.conv2(self.conv1(x))

class FPN_PAN(nn.Module):
    """Feature Pyramid Network (FPN) + Path Aggregation Network (PAN) untuk mengolah feature maps dari backbone dan menggabungkan fitur dari berbagai skala."""
    
    def __init__(self, in_channels: List[int], out_channels: Optional[List[int]] = None, logger: Optional[SmartCashLogger] = None):
        """Inisialisasi FPN-PAN neck dengan in_channels dari backbone [C3, C4, C5] dan out_channels opsional [P3, P4, P5]."""
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.in_channels, self.out_channels = in_channels, out_channels or YOLO_CHANNELS
        
        # Validasi input dan output channels
        if len(self.in_channels) != 3: raise NeckError(f"❌ FPN-PAN membutuhkan 3 input feature maps, tetapi {len(self.in_channels)} diberikan")
        if len(self.out_channels) != 3: raise NeckError(f"❌ FPN-PAN membutuhkan 3 output feature maps, tetapi {len(self.out_channels)} diberikan")
        
        # Inisialisasi FPN dan PAN
        self.fpn = FeaturePyramidNetwork(in_channels=in_channels, out_channels=self.out_channels)
        self.pan = PathAggregationNetwork(in_channels=self.out_channels, out_channels=self.out_channels)
        self.logger.info(f"✨ FPN-PAN neck diinisialisasi:\n   • Input channels: {in_channels}\n   • Output channels: {self.out_channels}")
        
    def get_output_channels(self) -> List[int]:
        """Mengembalikan daftar output channels dari neck."""
        return self.out_channels
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass FPN-PAN neck dengan input feature maps dari backbone [C3, C4, C5]."""
        try:
            # Validasi input dan proses melalui FPN dan PAN
            if len(features) != 3: raise NeckError(f"❌ FPN-PAN membutuhkan 3 input feature maps, tetapi {len(features)} diberikan")
            if self.logger: self.logger.debug(f"🔍 Input feature shapes: {[f.shape for f in features]}")
            
            # Proses features melalui FPN dan kemudian PAN
            fpn_features = self.fpn(features)
            pan_features = self.pan(fpn_features)
            
            if self.logger: self.logger.debug(f"🔍 Output shapes: FPN {[f.shape for f in fpn_features]}, PAN {[f.shape for f in pan_features]}")
            return pan_features
        except Exception as e:
            self.logger.error(f"❌ FPN-PAN forward pass gagal: {str(e)}")
            raise NeckError(f"Forward pass gagal: {str(e)}")

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network yang disempurnakan dengan residual blocks untuk mengkombinasikan fitur dari berbagai level dengan koneksi top-down untuk menambahkan informasi semantik ke fitur resolusi tinggi."""
    
    def __init__(self, in_channels: List[int], out_channels: List[int]):
        """Inisialisasi Feature Pyramid Network."""
        super().__init__()
        # Lateral connections dan residual blocks
        self.lateral_convs = nn.ModuleList([ConvBlock(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(in_channels, out_channels)])
        self.fpn_blocks = nn.ModuleList([nn.Sequential(*[ResidualBlock(out_ch) for _ in range(3)]) for out_ch in out_channels])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass FPN dengan input features dari backbone [P3, P4, P5]."""
        # Lateral connections dan top-down pathway
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        fpn_features = [self.fpn_blocks[-1](laterals[-1])]  # Start dengan level tertinggi (P5)
        
        # Lakukan koneksi top-down dengan iterasi balik
        for i in range(len(laterals)-2, -1, -1):
            # Upsample fitur level atas
            top_down = self.upsample(fpn_features[0])
            
            # Pastikan channel top_down sesuai dengan laterals[i] untuk operasi penjumlahan
            if top_down.shape[1] != laterals[i].shape[1]:
                # Jika channel tidak sesuai, gunakan konvolusi 1x1 untuk menyesuaikan channel
                adapter = nn.Conv2d(top_down.shape[1], laterals[i].shape[1], kernel_size=1).to(top_down.device)
                top_down = adapter(top_down)
            
            # Tambahkan lateral connection dan apply residual blocks
            fpn_feature = self.fpn_blocks[i](laterals[i] + top_down)
            fpn_features.insert(0, fpn_feature)  # Insert di awal list untuk urutan (P3, P4, P5)
            
        return fpn_features

class PathAggregationNetwork(nn.Module):
    """Path Aggregation Network untuk bottom-up path augmentation yang memperkuat propagasi fitur lokal ke level yang lebih tinggi."""
    
    def __init__(self, in_channels: List[int], out_channels: List[int], num_repeats: int = 3):
        """Inisialisasi Path Aggregation Network dengan in_channels dari FPN, out_channels yang diinginkan, dan jumlah residual blocks."""
        super().__init__()
        # Downsampling, bottom-up convolutions, dan residual blocks
        self.downsamples = nn.ModuleList([ConvBlock(in_ch, in_ch, kernel_size=3, stride=2) for in_ch in in_channels[:-1]])
        self.bu_convs = nn.ModuleList([ConvBlock(in_ch*2, out_ch, kernel_size=1) for in_ch, out_ch in zip(in_channels[1:], out_channels[1:])])
        self.pan_blocks = nn.ModuleList([nn.Sequential(*[ResidualBlock(out_ch) for _ in range(num_repeats)]) for out_ch in out_channels])
        
    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass PAN dengan input fpn_features dari FPN [P3, P4, P5]."""
        # Bottom-up pathway
        pan_features = [self.pan_blocks[0](fpn_features[0])]  # Start dengan level terendah (P3)
        
        # Proses bottom-up pathway
        for i in range(len(fpn_features)-1):
            # Downsample, concat, dan apply residual blocks
            bottom_up = self.downsamples[i](pan_features[-1])
            
            # Pastikan channel sesuai sebelum concat
            if bottom_up.shape[1] + fpn_features[i+1].shape[1] != self.bu_convs[i].conv.in_channels:
                # Jika channel tidak sesuai, gunakan adapter untuk menyesuaikan channel
                adapter = nn.Conv2d(
                    bottom_up.shape[1] + fpn_features[i+1].shape[1], 
                    self.bu_convs[i].conv.in_channels, 
                    kernel_size=1
                ).to(bottom_up.device)
                combined = torch.cat([bottom_up, fpn_features[i+1]], dim=1)
                combined = adapter(combined)
            else:
                combined = torch.cat([bottom_up, fpn_features[i+1]], dim=1)
            
            # Apply residual blocks
            pan_features.append(self.pan_blocks[i+1](self.bu_convs[i](combined)))
            
        return pan_features