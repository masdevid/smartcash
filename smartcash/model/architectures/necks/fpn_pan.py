"""
File: smartcash/model/architectures/necks/fpn_pan.py
Deskripsi: Implementasi Feature Pyramid Network dan Path Aggregation Network untuk pemrosesan dan fusi fitur multi-skala
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from smartcash.common.logger import SmartCashLogger
from smartcash.model.exceptions import NeckError

class FeatureProcessingNeck(nn.Module):
    """
    Feature Processing Neck mengkombinasikan Feature Pyramid Network (FPN) 
    dan Path Aggregation Network (PAN) untuk memproses dan menggabungkan 
    fitur dari berbagai skala secara optimal.
    """
    
    def __init__(
        self,
        in_channels: List[int],  # Channel dari backbone stages
        out_channels: List[int] = [128, 256, 512],  # Target output channels untuk YOLOv5
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi FeatureProcessingNeck.
        
        Args:
            in_channels: List jumlah channel dari backbone stages
            out_channels: List jumlah channel output yang diinginkan
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
            out_channels=out_channels
        )
        
        # Path Aggregation Network
        self.pan = PathAggregationNetwork(
            in_channels=out_channels,
            out_channels=out_channels
        )
        
        self.logger.info(
            f"✅ FeatureProcessingNeck diinisialisasi:\n"
            f"   • Input channels: {in_channels}\n"
            f"   • Output channels: {out_channels}"
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
            
            # FPN pass
            fpn_features = self.fpn(features)
            
            # Log dimensi output FPN untuk debugging
            if self.logger:
                self.logger.debug(f"🔍 FPN output shapes: {[f.shape for f in fpn_features]}")
            
            # PAN pass
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
    Feature Pyramid Network untuk mengkombinasikan fitur dari berbagai level
    dengan koneksi top-down untuk menambahkan informasi semantik ke fitur resolusi tinggi.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int]
    ):
        """
        Inisialisasi Feature Pyramid Network.
        
        Args:
            in_channels: List jumlah channel untuk setiap level backbone
            out_channels: List jumlah channel output yang diinginkan
        """
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
        Forward pass FPN.
        
        Args:
            features: List fitur dari backbone [P3, P4, P5]
            
        Returns:
            List[torch.Tensor]: Feature yang telah diproses FPN
        """
        # Lateral connections (reordering untuk mengikuti konvensi FPN)
        laterals = [
            conv(feature)
            for feature, conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway mulai dari feature level tertinggi
        fpn_features = [laterals[-1]]  # Start dengan level tertinggi
        
        # Lakukan koneksi top-down dengan iterasi balik
        for i in range(len(laterals)-2, -1, -1):
            # Upsample fitur level atas
            top_down = self.upsample(fpn_features[0])
            
            # Tambahkan dengan lateral connection
            fpn_feature = laterals[i] + top_down
            
            # Aplikasikan 3x3 conv untuk refine
            fpn_feature = self.fpn_convs[i](fpn_feature)
            
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
        out_channels: List[int]
    ):
        """
        Inisialisasi Path Aggregation Network.
        
        Args:
            in_channels: List jumlah channel dari FPN
            out_channels: List jumlah channel output yang diinginkan
        """
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
        Forward pass PAN.
        
        Args:
            fpn_features: List fitur dari FPN [P3, P4, P5]
            
        Returns:
            List[torch.Tensor]: Feature yang telah diproses PAN
        """
        # Bottom-up pathway mulai dari feature level terendah
        pan_features = [fpn_features[0]]  # Start dengan level terendah (P3)
        
        # Bottom-up pathway
        for i in range(len(fpn_features)-1):
            # Downsample fitur level bawah
            bottom_up = self.downsample(pan_features[-1])
            
            # Concat dengan fpn feature
            combined = torch.cat([bottom_up, fpn_features[i+1]], dim=1)
            
            # Aplikasikan conv
            pan_feature = self.bu_convs[i](combined)
            
            # Tambahkan ke list feature
            pan_features.append(pan_feature)
            
        return pan_features