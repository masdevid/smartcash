# File: models/backbones/efficientnet.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi EfficientNet backbone untuk YOLOv5

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import timm

from .base import BaseBackbone
from smartcash.utils.logger import SmartCashLogger

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B4 backbone implementation."""
    
    def __init__(self, pretrained: bool = True, logger: Optional[SmartCashLogger] = None):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        try:
            # Load pretrained EfficientNet-B4
            self.model = timm.create_model('efficientnet_b4', 
                                         pretrained=pretrained,
                                         features_only=True,
                                         out_indices=(2, 3, 4))  # P3, P4, P5 stages
            self.logger.info("✅ Loaded pretrained EfficientNet-B4 backbone")
        except Exception as e:
            self.logger.error(f"❌ Failed to load EfficientNet-B4: {str(e)}")
            raise
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for P3, P4, P5."""
        return [56, 160, 448]  # EfficientNet-B4 channels
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get output shapes for feature maps."""
        return [(80, 80), (40, 40), (20, 20)]  # For 640x640 input
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through EfficientNet backbone."""
        return self.model(x)