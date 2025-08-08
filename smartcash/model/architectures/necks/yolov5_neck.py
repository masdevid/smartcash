import torch
import torch.nn as nn
from typing import List
from ultralytics.nn.modules import C2f, Concat, Conv
from smartcash.model.architectures.necks.neck import BaseNeck
from smartcash.common.logger import get_logger

class YOLOv5Neck(BaseNeck):
    """YOLOv5 neck (PANet) implementation"""
    
    def __init__(self, feature_dims: List[int]):
        super().__init__(feature_dims)
        self.logger = get_logger(__name__)
        self.logger.info("✅ Created YOLOv5Neck")
    
    def create_neck(self) -> nn.ModuleList:
        return nn.ModuleList([
            Conv(self.feature_dims[2], self.feature_dims[1], 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C2f(self.feature_dims[1] + self.feature_dims[1], self.feature_dims[1], 1, False),
            Conv(self.feature_dims[1], self.feature_dims[0], 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C2f(self.feature_dims[0] + self.feature_dims[0], self.feature_dims[0], 1, False),
            Conv(self.feature_dims[0], self.feature_dims[0], 3, 2),
            Concat(1),
            C2f(self.feature_dims[0] + self.feature_dims[1], self.feature_dims[1], 1, False),
            Conv(self.feature_dims[1], self.feature_dims[1], 3, 2),
            Concat(1),
            C2f(self.feature_dims[1] + self.feature_dims[1], self.feature_dims[2], 1, False)
        ])
    
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> List[torch.Tensor]:
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
        return [p3_out, p4_final, p5_final]