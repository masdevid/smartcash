# File: models/yolov5_efficient.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi YOLOv5 dengan EfficientNet-B4 backbone untuk deteksi nilai mata uang

import torch
import torch.nn as nn
from typing import List, Optional
from utils.logger import SmartCashLogger
from models.backbones.efficientnet_backbone import EfficientNetBackbone

class YOLOv5Efficient(nn.Module):
    """
    Implementasi YOLOv5 dengan EfficientNet-B4 sebagai backbone.
    Dioptimasi untuk deteksi nilai mata uang Rupiah.
    """
    
    def __init__(
        self,
        num_classes: int,
        logger: Optional[SmartCashLogger] = None,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.num_classes = num_classes
        
        # Inisialisasi backbone
        self.logger.info("🔄 Inisialisasi EfficientNet-B4 backbone...")
        self.backbone = EfficientNetBackbone(
            logger=self.logger,
            pretrained=pretrained_backbone,
            trainable_layers=trainable_backbone_layers
        )
        
        # Feature Pyramid Network (neck)
        self.logger.info("🔄 Setup Feature Pyramid Network...")
        self.fpn = self._build_fpn()
        
        # YOLOv5 detection heads
        self.logger.info("🔄 Setup Detection Heads...")
        self.heads = self._build_heads()
        
    def _build_fpn(self) -> nn.ModuleList:
        """
        Membangun Feature Pyramid Network untuk feature fusion
        Returns:
            ModuleList berisi layer FPN
        """
        fpn_layers = []
        # Channel dimensi dari EfficientNet-B4 feature maps
        channels = [56, 160, 448]  # sesuai dengan block 2, 4, dan 6
        
        # Top-down pathway
        for i in range(len(channels)-1, 0, -1):
            fpn_layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i-1], 1),
                nn.BatchNorm2d(channels[i-1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2)
            ))
            
        # Bottom-up pathway
        for i in range(len(channels)-1):
            fpn_layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ))
            
        return nn.ModuleList(fpn_layers)
        
    def _build_heads(self) -> nn.ModuleList:
        """
        Membangun detection heads untuk setiap skala
        Returns:
            ModuleList berisi detection heads
        """
        heads = []
        channels = [56, 160, 448]  # Channel dimensi untuk setiap skala
        
        for ch in channels:
            heads.append(nn.Sequential(
                # Detection conv layers
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                
                # Output conv - [x, y, w, h, conf, num_classes]
                nn.Conv2d(ch, 5 + self.num_classes, 1)
            ))
            
        return nn.ModuleList(heads)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor [batch_size, channels, height, width]
        Returns:
            List predictions untuk setiap skala
        """
        # Backbone forward pass
        features = self.backbone(x)
        
        # FPN feature fusion
        fpn_features = []
        
        # Top-down pathway
        prev_feature = features[-1]
        fpn_features.insert(0, prev_feature)
        
        for i, fpn_layer in enumerate(self.fpn[:len(features)-1]):
            prev_feature = fpn_layer(prev_feature)
            prev_feature = torch.cat([prev_feature, features[-(i+2)]], dim=1)
            fpn_features.insert(0, prev_feature)
            
        # Detection heads
        predictions = []
        for feature, head in zip(fpn_features, self.heads):
            predictions.append(head(feature))
            
        return predictions
    
    def load_yolov5_weights(self, weights_path: str) -> None:
        """
        Load YOLOv5 pre-trained weights untuk head layers
        Args:
            weights_path: Path ke YOLOv5 weights
        """
        self.logger.info(f"📥 Loading YOLOv5 weights dari {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=self.backbone.device)
            # Filter dan load weights yang sesuai dengan head layers
            head_state_dict = {
                k: v for k, v in state_dict.items()
                if 'head' in k
            }
            self.heads.load_state_dict(head_state_dict, strict=False)
            self.logger.success("✨ YOLOv5 weights berhasil dimuat")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat weights: {str(e)}")
            raise