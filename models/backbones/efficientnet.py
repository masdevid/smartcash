# File: models/backbones/efficientnet.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi EfficientNet backbone untuk YOLOv5 dengan perbaikan output channels

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import timm

from .base import BaseBackbone
from smartcash.utils.logger import SmartCashLogger

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B4 backbone implementation dengan adaptasi untuk YOLOv5."""
    
    def __init__(self, pretrained: bool = True, logger: Optional[SmartCashLogger] = None):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        try:
            # Load pretrained EfficientNet-B4
            self.model = timm.create_model('efficientnet_b4', 
                                         pretrained=pretrained,
                                         features_only=True,
                                         out_indices=(2, 3, 4))  # P3, P4, P5 stages
            
            # Adapter layers to convert EfficientNet output channels to YOLO expected channels
            # EfficientNet-B4 outputs: [56, 160, 448]
            # YOLOv5 expects: [128, 256, 512]
            self.adapters = nn.ModuleList([
                nn.Conv2d(56, 128, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(448, 512, kernel_size=1, stride=1, padding=0)
            ])
            
            self.logger.info("✅ Loaded pretrained EfficientNet-B4 backbone with channel adapters")
        except Exception as e:
            self.logger.error(f"❌ Failed to load EfficientNet-B4: {str(e)}")
            raise
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for P3, P4, P5."""
        # Return adapted channel sizes to match YOLOv5 expectations
        return [128, 256, 512]
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get output shapes for feature maps."""
        return [(80, 80), (40, 40), (20, 20)]  # For 640x640 input
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through EfficientNet backbone with channel adaptation."""
        # Get multi-scale features from EfficientNet
        features = self.model(x)
        
        # Apply channel adapters
        adapted_features = []
        for feat, adapter in zip(features, self.adapters):
            adapted_features.append(adapter(feat))
            
        return adapted_features