# File: models/yolov5_efficient.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi model YOLOv5 dengan EfficientNet backbone dan FPN+PAN neck

import torch
import torch.nn as nn
from typing import List, Optional

from .backbones.efficient_adapter import EfficientNetAdapter
from .necks.fpn_pan import FeatureProcessingNeck
from utils.logger import SmartCashLogger

class YOLOv5Efficient(nn.Module):
    """
    Implementasi YOLOv5 dengan EfficientNet-B4 backbone untuk deteksi nilai mata uang
    """
    
    def __init__(
        self,
        num_classes: int = 7,  # 7 denominasi Rupiah
        pretrained: bool = True,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.num_classes = num_classes
        
        # Inisialisasi backbone dengan EfficientNet-B4
        self.backbone = EfficientNetAdapter(
            pretrained=pretrained,
            trainable_layers=3
        )
        
        # Inisialisasi FPN+PAN neck
        self.neck = FeatureProcessingNeck(
            in_channels=[56, 160, 448],  # EfficientNet channels
            out_channels=[128, 256, 512]  # YOLOv5 channels
        )
        
        # Detection head untuk setiap skala
        self.detection_heads = nn.ModuleList([
            DetectionHead(
                in_channels=ch,
                num_classes=num_classes
            ) for ch in [128, 256, 512]
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass model
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            List prediksi dari detection heads
        """
        # Extract features dari backbone
        backbone_features = self.backbone(x)
        
        # Proses feature dengan FPN+PAN
        neck_features = self.neck(backbone_features)
        
        # Deteksi pada setiap skala
        predictions = []
        for feature, head in zip(neck_features, self.detection_heads):
            pred = head(feature)
            predictions.append(pred)
            
        return predictions

class DetectionHead(nn.Module):
    """
    Detection head untuk mata uang dengan format output yang disesuaikan
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 3
    ):
        super().__init__()
        
        # Convolution untuk deteksi
        self.conv = nn.Sequential(
            self._conv_block(in_channels, in_channels//2),
            self._conv_block(in_channels//2, in_channels//2),
            nn.Conv2d(
                in_channels//2,
                num_anchors * (5 + num_classes),  # 5: x,y,w,h,obj
                kernel_size=1
            )
        )
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def _conv_block(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1
    ) -> nn.Sequential:
        """Helper untuk membuat conv block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass detection head
        Args:
            x: Input feature map
        Returns:
            Prediksi dalam format [batch, anchors*(5+classes), h, w]
        """
        batch_size = x.shape[0]
        
        # Aplikasikan convolution
        x = self.conv(x)
        
        # Reshape output
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, x.shape[-2], x.shape[-1])
        x = x.permute(0, 1, 3, 4, 2)  # [b, anchors, h, w, 5+classes]
        
        return x