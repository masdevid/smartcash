# File: models/backbones/cspdarknet.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi CSPDarknet backbone dari YOLOv5

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .base import BaseBackbone
from smartcash.utils.logger import SmartCashLogger

class CSPDarknet(BaseBackbone):
    """CSPDarknet backbone implementation."""
    
    def __init__(self, pretrained: bool = True, logger: Optional[SmartCashLogger] = None):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        # Initialize YOLOv5 CSPDarknet layers
        try:
            import torch.hub
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)
            self.backbone = self.model.model.model[:10]  # Extract backbone layers
            self.logger.info("✅ Loaded pretrained YOLOv5 CSPDarknet backbone")
        except Exception as e:
            self.logger.error(f"❌ Failed to load YOLOv5: {str(e)}")
            raise
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for P3, P4, P5."""
        return [128, 256, 512]  # CSPDarknet channels
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get output shapes for feature maps."""
        return [(80, 80), (40, 40), (20, 20)]  # For 640x640 input
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through CSPDarknet backbone."""
        features = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in [4, 6, 9]:  # P3, P4, P5 layers
                features.append(x)
        return features