import torch
import torch.nn as nn
from typing import List
from ultralytics.nn.modules import Detect
from smartcash.model.architectures.heads.head import BaseHead
from smartcash.common.logger import get_logger

class YOLOv5Head(BaseHead):
    """YOLOv5 detection head implementation"""
    
    def __init__(self, num_classes: int, feature_dims: List[int]):
        super().__init__(num_classes, feature_dims)
        self.logger = get_logger(__name__)
        self.logger.info("✅ Created YOLOv5Head")
    
    def create_head(self) -> nn.Module:
        return Detect(nc=self.num_classes, ch=self.feature_dims)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return self.head(features)