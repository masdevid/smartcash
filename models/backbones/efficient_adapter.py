# File: models/backbones/efficient_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Update EfficientNet adapter untuk integrasi dengan FPN

import torch
import torch.nn as nn
import timm
from typing import List, Optional
from utils.logger import SmartCashLogger

class EfficientNetAdapter(nn.Module):
    """Adapter untuk integrasi EfficientNet-B4 dengan FPN+PAN"""
    
    def __init__(
        self,
        pretrained: bool = True,
        trainable_layers: int = 3,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        # Load EfficientNet-B4 menggunakan timm
        self.logger.info("🔄 Loading EfficientNet-B4 backbone...")
        self.efficientnet = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=(2, 3, 4)  # P3, P4, P5 stages
        )
        
        # Channel dimensions untuk setiap stage
        self.channels = self.efficientnet.feature_info.channels()
        self.logger.info(f"📊 Feature channels: {self.channels}")
        
        # Freeze layers sesuai parameter
        self._freeze_layers(trainable_layers)
        
    def _freeze_layers(self, trainable_layers: int):
        """Freeze backbone layers yang tidak ditraining"""
        if trainable_layers < 5:  # Total 5 stages
            layers_to_freeze = len(self.efficientnet.blocks) - trainable_layers
            for i, block in enumerate(self.efficientnet.blocks[:layers_to_freeze]):
                self.logger.info(f"❄️ Freezing layer {i}")
                for param in block.parameters():
                    param.requires_grad = False
                        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass dengan multi-scale feature extraction
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            List feature maps [P3, P4, P5]
        """
        # Extract features menggunakan timm's features_only mode
        features = self.efficientnet(x)
        
        # Log shapes untuk debugging
        self.logger.debug("Feature shapes:")
        for i, feat in enumerate(features):
            self.logger.debug(f"P{i+3}: {feat.shape}")
                
        return features
        
    @property
    def feature_channels(self) -> List[int]:
        """Getter untuk channel dimensions dari setiap stage"""
        return self.channels