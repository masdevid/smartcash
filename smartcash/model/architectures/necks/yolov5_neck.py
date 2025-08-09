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
            # Top-down pathway (simplified)
            Conv(self.feature_dims[2], self.feature_dims[1], 1, 1),  # P5 -> 256
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            Conv(self.feature_dims[1] + self.feature_dims[1], self.feature_dims[1], 3, 1),  # 512 -> 256
            Conv(self.feature_dims[1], self.feature_dims[0], 1, 1),  # 256 -> 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),  
            Conv(self.feature_dims[0] + self.feature_dims[0], self.feature_dims[0], 3, 1),  # 256 -> 128
            # Bottom-up pathway (simplified)  
            Conv(self.feature_dims[0], self.feature_dims[0], 3, 2),  # 128 -> 128
            Concat(1),
            Conv(self.feature_dims[0] + self.feature_dims[1], self.feature_dims[1], 3, 1),  # 128 + 256 -> 256
            Conv(self.feature_dims[1], self.feature_dims[1], 3, 2),  # 256 -> 256
            Concat(1),
            Conv(self.feature_dims[1] + self.feature_dims[1], self.feature_dims[2], 3, 1)  # 256 + 256 -> 512
        ])
    
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> List[torch.Tensor]:
        # OPTIMIZATION: Simplified forward pass matching lightweight neck
        # Top-down pathway
        p5_conv = self.neck[0](p5)          # P5 -> reduced
        p5_up = self.neck[1](p5_conv)       # Upsample
        p4_cat = self.neck[2]([p5_up, p4])  # Concat with P4
        p4_out = self.neck[3](p4_cat)       # Fuse P4
        
        p4_conv = self.neck[4](p4_out)      # P4 -> reduced  
        p4_up = self.neck[5](p4_conv)       # Upsample
        p3_cat = self.neck[6]([p4_up, p3])  # Concat with P3
        p3_out = self.neck[7](p3_cat)       # Fuse P3
        
        # Bottom-up pathway
        p3_down = self.neck[8](p3_out)      # P3 -> P4
        p4_cat2 = self.neck[9]([p3_down, p4_out])  # Concat
        p4_final = self.neck[10](p4_cat2)   # Fuse P4
        
        p4_down = self.neck[11](p4_final)   # P4 -> P5
        p5_cat = self.neck[12]([p4_down, p5_conv])  # Concat
        p5_final = self.neck[13](p5_cat)    # Fuse P5
        
        return [p3_out, p4_final, p5_final]