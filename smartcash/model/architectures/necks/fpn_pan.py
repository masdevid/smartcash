"""
File: smartcash/model/architectures/necks/fpn_pan.py
Deskripsi: Feature Processing Neck implementation for YOLOv5
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import NeckError

class ConvBlock(nn.Module):
    """Convolution block dengan BatchNorm dan aktivasi."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: Optional[int] = None
    ):
        """
        Inisialisasi Conv Block.
        
        Args:
            in_channels: Jumlah channel input
            out_channels: Jumlah channel output
            kernel_size: Ukuran kernel
            stride: Stride konvolusi
            padding: Padding (None untuk auto-padding)
        """
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # Same padding
            
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass Conv Block."""
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block untuk mempertahankan informasi."""
    
    def __init__(self, channels: int):
        """
        Inisialisasi Residual Block.
        
        Args:
            channels: Jumlah channel
        """
        super().__init__()
        mid_channels = channels // 2
        
        self.conv1 = ConvBlock(channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBlock(mid_channels, channels, kernel_size=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dengan residual connection."""
        return x + self.conv2(self.conv1(x))

class FeatureProcessingNeck(nn.Module):
    """
    Feature Processing Neck mengkombinasikan Feature Pyramid Network (FPN) 
    dan Path Aggregation Network (PAN) dengan residual blocks untuk pemrosesan
    dan fusi fitur dari berbagai skala secara optimal.
    """
    
    def __init__(
        self,
        in_channels: List[int],  # Channel dari backbone stages
        out_channels: List[int] = [128, 256, 512],  # Target output channels untuk YOLOv5
        num_repeats: int = 3,    # Jumlah residual blocks
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi FeatureProcessingNeck.
        
        Args:
            in_channels: List jumlah channel dari backbone stages
            out_channels: List jumlah channel output yang diinginkan
            num_repeats: Jumlah residual blocks untuk feature enhancement
            logger: Logger untuk mencatat proses (opsional)
            
        Raises:
            NeckError: Jika parameter tidak valid
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        # Validasi parameter
        if len(in_channels) != len(out_channels):
            raise NeckError(
                f"❌ Jumlah in_channels ({len(in_channels)}) harus sama dengan "
                f"jumlah out_channels ({len(out_channels)})"
            )
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_repeats=num_repeats
        )
        
        # Path Aggregation Network
        self.pan = PathAggregationNetwork(
            in_channels=out_channels,
            out_channels=out_channels,
            num_repeats=num_repeats
        )
        
        self.logger.info(
            f"✅ FeatureProcessingNeck diinisialisasi:\n"
            f"   • Input channels: {in_channels}\n"
            f"   • Output channels: {out_channels}\n"
            f"   • Residual blocks: {num_repeats}"
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass FPN-PAN.
        
        Args:
            features: List fitur dari backbone [P3, P4, P5]
            
        Returns:
            List[torch.Tensor]: Feature yang telah diproses
            
        Raises:
            NeckError: Jika forward pass gagal
        """
        try:
            # Validasi input features
            if not features or len(features) < 3:
                raise NeckError(
                    f"❌ Jumlah feature maps ({len(features) if features else 0}) "
                    f"tidak mencukupi, minimal diperlukan 3 feature maps"
                )
            
            # Log dimensi input untuk debugging
            if self.logger:
                self.logger.debug(f"🔍 Input feature shapes: {[f.shape for f in features]}")
            
            # FPN pass (Top-down pathway)
            fpn_features = self.fpn(features)
            
            # Log dimensi output FPN untuk debugging
            if self.logger:
                self.logger.debug(f"🔍 FPN output shapes: {[f.shape for f in fpn_features]}")
            
            # PAN pass (Bottom-up pathway)
            pan_features = self.pan(fpn_features)
            
            # Log dimensi output PAN untuk debugging
            if self.logger:
                self.logger.debug(f"🔍 Final output shapes: {[f.shape for f in pan_features]}")
            
            return pan_features
            
        except Exception as e:
            self.logger.error(f"❌ FeatureProcessingNeck forward pass gagal: {str(e)}")
            raise NeckError(f"Forward pass gagal: {str(e)}")
        
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network yang disempurnakan dengan residual blocks
    untuk mengkombinasikan fitur dari berbagai level dengan koneksi 
    top-down untuk menambahkan informasi semantik ke fitur resolusi tinggi.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        num_repeats: int = 3
    ):
        """
        Inisialisasi Feature Pyramid Network.
        
        Args:
            in_channels: List jumlah channel untuk setiap level backbone
            out_channels: List jumlah channel output yang diinginkan
            num_repeats: Jumlah residual blocks
        """
        super().__init__()
        
        # Lateral connections (1x1 conv untuk menyesuaikan channel)
        self.lateral_convs = nn.ModuleList([
            ConvBlock(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        
        # Residual blocks untuk feature enhancement
        self.fpn_blocks = nn.ModuleList([
            nn.Sequential(*[
                ResidualBlock(out_ch) for _ in range(num_repeats)
            ]) for out_ch in out_channels
        ])
        
        # Upsampling untuk top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass FPN.
        
        Args:
            features: List fitur dari backbone [P3, P4, P5]
            
        Returns:
            List[torch.Tensor]: Feature yang telah diproses FPN
        """
        # Lateral connections untuk menyesuaikan channel
        laterals = [
            conv(feature)
            for feature, conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway mulai dari feature level tertinggi
        fpn_features = [laterals[-1]]  # Start dengan level tertinggi (P5)
        fpn_features[0] = self.fpn_blocks[-1](fpn_features[0])  # Apply residual blocks
        
        # Lakukan koneksi top-down dengan iterasi balik
        for i in range(len(laterals)-2, -1, -1):
            # Upsample fitur level atas
            top_down = self.upsample(fpn_features[0])
            
            # Tambahkan dengan lateral connection
            fpn_feature = laterals[i] + top_down
            
            # Apply residual blocks untuk enhancement
            fpn_feature = self.fpn_blocks[i](fpn_feature)
            
            # Insert di awal list untuk mempertahankan urutan (P3, P4, P5)
            fpn_features.insert(0, fpn_feature)
            
        return fpn_features

class PathAggregationNetwork(nn.Module):
    """
    Path Aggregation Network untuk bottom-up path augmentation
    yang memperkuat propagasi fitur lokal ke level yang lebih tinggi.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        num_repeats: int = 3
    ):
        """
        Inisialisasi Path Aggregation Network.
        
        Args:
            in_channels: List jumlah channel dari FPN
            out_channels: List jumlah channel output yang diinginkan
            num_repeats: Jumlah residual blocks
        """
        super().__init__()
        
        # Downsampling layers untuk bottom-up pathway
        self.downsamples = nn.ModuleList([
            ConvBlock(in_ch, in_ch, kernel_size=3, stride=2)
            for in_ch in in_channels[:-1]
        ])
        
        # Bottom-up convolutions dengan concatenation
        self.bu_convs = nn.ModuleList([
            ConvBlock(in_ch*2, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(in_channels[1:], out_channels[1:])
        ])
        
        # Residual blocks untuk feature enhancement
        self.pan_blocks = nn.ModuleList([
            nn.Sequential(*[
                ResidualBlock(out_ch) for _ in range(num_repeats)
            ]) for out_ch in out_channels
        ])
        
    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass PAN.
        
        Args:
            fpn_features: List fitur dari FPN [P3, P4, P5]
            
        Returns:
            List[torch.Tensor]: Feature yang telah diproses PAN
        """
        # Bottom-up pathway mulai dari feature level terendah
        pan_features = [fpn_features[0]]  # Start dengan level terendah (P3)
        pan_features[0] = self.pan_blocks[0](pan_features[0])  # Apply residual blocks
        
        # Bottom-up pathway
        for i in range(len(fpn_features)-1):
            # Downsample fitur level bawah
            bottom_up = self.downsamples[i](pan_features[-1])
            
            # Concat dengan fpn feature
            combined = torch.cat([bottom_up, fpn_features[i+1]], dim=1)
            
            # Apply 1x1 conv untuk channel reduction
            pan_feature = self.bu_convs[i](combined)
            
            # Apply residual blocks untuk enhancement
            pan_feature = self.pan_blocks[i+1](pan_feature)
            
            # Tambahkan ke list feature
            pan_features.append(pan_feature)
            
        return pan_features