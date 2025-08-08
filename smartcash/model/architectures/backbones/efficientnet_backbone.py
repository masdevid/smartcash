import torch
import torch.nn as nn
import timm
from typing import List
from ultralytics.nn.modules import C2f, SPPF, Concat, Detect
from smartcash.model.architectures.backbones.backbone import BaseBackbone
from smartcash.common.logger import get_logger

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B4 backbone implementation"""
    
    def __init__(self, num_classes: int, pretrained: bool, device: str):
        super().__init__(num_classes, pretrained, device)
        self.logger = get_logger(__name__)
        self.backbone = self.create_backbone()
        self.feature_dims = [56, 160, 448]
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
    
    def create_neck(self, feature_dims: List[int]) -> nn.ModuleList:
        return nn.ModuleList([
            SPPF(feature_dims[2], feature_dims[2], 5),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C2f(feature_dims[2] + feature_dims[1], feature_dims[1], 1, False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C2f(feature_dims[1] + feature_dims[0], feature_dims[0], 1, False),
            nn.Conv2d(feature_dims[0], feature_dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dims[0]),
            nn.SiLU(),
            Concat(1),
            C2f(feature_dims[0] + feature_dims[1], feature_dims[1], 1, False),
            nn.Conv2d(feature_dims[1], feature_dims[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dims[1]),
            nn.SiLU(),
            Concat(1),
            C2f(feature_dims[1] + feature_dims[2], feature_dims[2], 1, False)
        ])
    
    def create_head(self, feature_dims: List[int]) -> nn.Module:
        return Detect(nc=self.num_classes, ch=feature_dims)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        p3, p4, p5 = features
        p5 = self.neck[0](p5)
        p5_up = self.neck[1](p5)
        p4_concat = self.neck[2]([p5_up, p4])
        p4_out = self.neck[3](p4_concat)
        p4_up = self.neck[4](p4_out)
        p3_concat = self.neck[5]([p4_up, p3])
        p3_out = self.neck[6](p3_concat)
        p3_down = self.neck[9](self.neck[8](self.neck[7](p3_out)))
        p4_concat2 = self.neck[10]([p3_down, p4_out])
        p4_final = self.neck[11](p4_concat2)
        p4_down = self.neck[14](self.neck[13](self.neck[12](p4_final)))
        p5_concat = self.neck[15]([p4_down, p5])
        p5_final = self.neck[16](p5_concat)
        return self.head([p3_out, p4_final, p5_final])