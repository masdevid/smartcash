import torch
import torch.nn as nn
from typing import List
from ultralytics.nn.modules import C2f, SPPF, Concat, Detect, Conv
from smartcash.model.architectures.backbones.backbone import BaseBackbone
from smartcash.common.logger import get_logger

class YOLOv5Backbone(BaseBackbone):
    """YOLOv5 backbone implementation"""
    
    def __init__(self, backbone: str, num_classes: int, pretrained: bool, device: str):
        super().__init__(num_classes, pretrained, device)
        self.logger = get_logger(__name__)
        self.backbone_type = backbone
        self.backbone = self.create_backbone()
        self.feature_dims = [128, 256, 512] if backbone == 'yolov5s' else [256, 512, 1024]
        self.neck = self.create_neck(self.feature_dims)
        self.head = self.create_head(self.feature_dims)
        self._setup_phase_1(self)
        self.to(self.device)
        self.logger.info(f"✅ Created YOLOv5Backbone: {backbone}, {num_classes} classes")
    
    def create_backbone(self) -> nn.ModuleList:
        channels = [32, 64, 128, 256, 512] if self.backbone_type == 'yolov5s' else [64, 128, 256, 512, 1024]
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
    
    def create_neck(self, feature_dims: List[int]) -> nn.ModuleList:
        return nn.ModuleList([
            Conv(feature_dims[2], feature_dims[1], 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C2f(feature_dims[1] + feature_dims[1], feature_dims[1], 1, False),
            Conv(feature_dims[1], feature_dims[0], 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C2f(feature_dims[0] + feature_dims[0], feature_dims[0], 1, False),
            Conv(feature_dims[0], feature_dims[0], 3, 2),
            Concat(1),
            C2f(feature_dims[0] + feature_dims[1], feature_dims[1], 1, False),
            Conv(feature_dims[1], feature_dims[1], 3, 2),
            Concat(1),
            C2f(feature_dims[1] + feature_dims[1], feature_dims[2], 1, False)
        ])
    
    def create_head(self, feature_dims: List[int]) -> nn.Module:
        return Detect(nc=self.num_classes, ch=feature_dims)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:
                features.append(x)
        
        p3, p4, p5 = features
        p5_conv = self.neck[0](p5)
        p5_up = self.neck[1](p5_conv)
        p4_cat = self.neck[2]([p5_up, p4])
        p4_out = self.neck[3](p4_cat)
        p4_conv = self.neck[4](p4_out)
        p4_up = self.neck[5](p4_conv)
        p3_cat = self.neck[6]([p4_up, p3])
        p3_out = self.neck[7](p3_cat)
        p3_down = self.neck[8](p3_out)
        p4_cat2 = self.neck[9]([p3_down, p4_out])
        p4_final = self.neck[10](p4_cat2)
        p4_down = self.neck[11](p4_final)
        p5_cat = self.neck[12]([p4_down, p5_conv])
        p5_final = self.neck[13](p5_cat)
        return self.head([p3_out, p4_final, p5_final])