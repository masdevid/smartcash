import torch
import torch.nn as nn
from typing import List
from smartcash.model.architectures.heads.head import BaseHead
from smartcash.common.logger import get_logger

class SmartCashDetectionHead(nn.Module):
    """Custom detection head that correctly handles 17 classes"""
    
    def __init__(self, num_classes: int, ch: List[int]):
        super().__init__()
        self.nc = num_classes  # Number of classes
        self.no = 3 * (num_classes + 5)  # 3 anchors * (classes + bbox_params)
        self.nl = len(ch)  # Number of detection layers
        
        # Detection layers - separate classification and regression heads
        self.cv2 = nn.ModuleList()  # Classification heads
        self.cv3 = nn.ModuleList()  # Regression heads
        
        for ch_i in ch:
            # Classification head: predicts class probabilities
            self.cv2.append(nn.Sequential(
                nn.Conv2d(ch_i, ch_i // 2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_i // 2, num_classes * 3, 1)  # 3 anchors * num_classes
            ))
            
            # Regression head: predicts bbox coordinates and objectness
            self.cv3.append(nn.Sequential(
                nn.Conv2d(ch_i, ch_i // 2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_i // 2, 5 * 3, 1)  # 3 anchors * (4 bbox + 1 objectness)
            ))
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass combining classification and regression outputs"""
        outputs = []
        
        for i in range(self.nl):
            # Get classification and regression predictions
            cls_pred = self.cv2[i](x[i])  # [batch, nc*3, h, w]
            reg_pred = self.cv3[i](x[i])  # [batch, 5*3, h, w]
            
            # Concatenate to form final prediction: [batch, (nc+5)*3, h, w]
            combined = torch.cat([reg_pred, cls_pred], dim=1)  # [batch, (5+nc)*3, h, w]
            outputs.append(combined)
        
        return outputs

class YOLOv5Head(BaseHead):
    """YOLOv5 detection head implementation with correct 17-class support"""
    
    def __init__(self, num_classes: int, feature_dims: List[int]):
        super().__init__(num_classes, feature_dims)
        self.logger = get_logger(__name__)
        self.logger.info("✅ Created YOLOv5Head")
    
    def create_head(self) -> nn.Module:
        return SmartCashDetectionHead(self.num_classes, self.feature_dims)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.head(features)