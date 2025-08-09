import torch
import torch.nn as nn
import timm
from typing import List
from .backbone import BaseBackbone
from smartcash.model.architectures.necks.yolov5_neck import YOLOv5Neck
from smartcash.model.architectures.heads.yolov5_head import YOLOv5Head
from smartcash.common.logger import SmartCashLogger

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B4 backbone implementation"""
    
    def __init__(self, num_classes: int, pretrained: bool, device: str):
        super().__init__(num_classes, pretrained, device)
        self.feature_dims = [56, 160, 448]
        self.backbone = self.create_backbone()
        self.neck = self.create_neck(self.feature_dims)
        self.head = self.create_head(self.feature_dims)
        self._setup_phase_1(self)
        self.to(self.device)
        self.logger.info(f"✅ Created EfficientNetBackbone: efficientnet_b4, {num_classes} classes")
    
    def create_backbone(self) -> nn.Module:
        if not timm:
            raise RuntimeError("timm is required for EfficientNet-B4 backbone")
        return timm.create_model(
            'efficientnet_b4',
            pretrained=self.pretrained,
            features_only=True,
            out_indices=[2, 3, 4]
        )
    
    def create_neck(self, feature_dims: List[int]) -> nn.Module:
        return YOLOv5Neck(feature_dims)
    
    def create_head(self, feature_dims: List[int]) -> nn.Module:
        return YOLOv5Head(self.num_classes, feature_dims)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        p3, p4, p5 = features
        neck_output = self.neck(p3, p4, p5)
        return self.head(neck_output)