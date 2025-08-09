import torch
import torch.nn as nn
from typing import List
from ultralytics.nn.modules import Conv, C2f, SPPF
from .backbone import BaseBackbone
from smartcash.model.architectures.necks.yolov5_neck import YOLOv5Neck
from smartcash.model.architectures.heads.yolov5_head import YOLOv5Head
from smartcash.common.logger import SmartCashLogger

class YOLOv5Backbone(BaseBackbone):
    """YOLOv5 backbone implementation"""
    
    def __init__(self, backbone: str, num_classes: int, pretrained: bool, device: str):
        super().__init__(num_classes, pretrained, device)
        self.backbone_type = backbone
        # OPTIMIZATION: Always use small variant channels for efficiency (<10M target)
        self.feature_dims = [128, 256, 512]  # Small variant for all backbones
        self.backbone = self.create_backbone()
        self.neck = self.create_neck(self.feature_dims)
        self.head = self.create_head(self.feature_dims)
        self._setup_phase_1(self)
        self.to(self.device)
        self.logger.info(f"✅ Created YOLOv5Backbone: {backbone}, {num_classes} classes")
    
    def create_backbone(self) -> nn.ModuleList:
        # OPTIMIZATION: Always use small channels for efficiency (<10M params target)
        channels = [32, 64, 128, 256, 512]  # Small variant for all backbone types
        return nn.ModuleList([
            Conv(3, channels[0], 6, 2, 2),
            Conv(channels[0], channels[1], 3, 2),
            C2f(channels[1], channels[1], 1, True),
            Conv(channels[1], channels[2], 3, 2),
            C2f(channels[2], channels[2], 2, True),
            Conv(channels[2], channels[3], 3, 2),
            C2f(channels[3], channels[3], 3, True),
            Conv(channels[3], channels[4], 3, 2),
            C2f(channels[4], channels[4], 1, True),
            SPPF(channels[4], channels[4], 5)
        ])
    
    def create_neck(self, feature_dims: List[int]) -> nn.Module:
        return YOLOv5Neck(feature_dims)
    
    def create_head(self, feature_dims: List[int]) -> nn.Module:
        return YOLOv5Head(self.num_classes, feature_dims)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:
                features.append(x)
        p3, p4, p5 = features
        neck_output = self.neck(p3, p4, p5)
        return self.head(neck_output)