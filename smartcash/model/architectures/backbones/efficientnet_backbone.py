import torch
import torch.nn as nn
import timm
from typing import List
from .backbone import BaseBackbone
from smartcash.model.architectures.necks.yolov5_neck import YOLOv5Neck
from smartcash.model.architectures.heads.yolov5_head import YOLOv5Head
from smartcash.common.logger import SmartCashLogger

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B4 backbone implementation (optimized neck/head to reduce bloat)"""
    
    def __init__(self, num_classes: int, pretrained: bool, device: str):
        super().__init__(num_classes, pretrained, device)
        # Keep EfficientNet-B4 backbone but optimize neck/head dimensions
        self.efficientnet_variant = 'efficientnet_b4'
        # EfficientNet-B4 natural output dimensions
        self.backbone_feature_dims = [56, 160, 448] 
        # OPTIMIZATION: Reduce dimensions passed to neck/head to cut parameters
        self.feature_dims = [28, 80, 224]  # Half the channels for neck/head
        self.backbone = self.create_backbone()
        # Add feature adaptation layers to reduce channels
        self.feature_adapters = self.create_feature_adapters()
        self.neck = self.create_neck(self.feature_dims)
        self.head = self.create_head(self.feature_dims)
        self._setup_phase_1(self)
        self.to(self.device)
        self.logger.info(f"✅ Created EfficientNetBackbone: {self.efficientnet_variant}, {num_classes} classes")
    
    def create_backbone(self) -> nn.Module:
        if not timm:
            raise RuntimeError("timm is required for EfficientNet backbone")
        return timm.create_model(
            self.efficientnet_variant,  # Use B0 instead of B4
            pretrained=self.pretrained,
            features_only=True,
            out_indices=[2, 3, 4]
        )
    
    def create_feature_adapters(self) -> nn.ModuleList:
        """Create 1x1 conv layers to reduce EfficientNet-B4 feature dimensions"""
        from ultralytics.nn.modules import Conv
        return nn.ModuleList([
            Conv(self.backbone_feature_dims[0], self.feature_dims[0], 1, 1),  # 56 -> 28
            Conv(self.backbone_feature_dims[1], self.feature_dims[1], 1, 1),  # 160 -> 80  
            Conv(self.backbone_feature_dims[2], self.feature_dims[2], 1, 1)   # 448 -> 224
        ])
    
    def create_neck(self, feature_dims: List[int]) -> nn.Module:
        return YOLOv5Neck(feature_dims)
    
    def create_head(self, feature_dims: List[int]) -> nn.Module:
        return YOLOv5Head(self.num_classes, feature_dims)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EfficientNet-B4 backbone forward pass
        features = self.backbone(x)
        p3, p4, p5 = features
        
        # OPTIMIZATION: Reduce feature dimensions for neck/head  
        p3_reduced = self.feature_adapters[0](p3)  # 56 -> 28 channels
        p4_reduced = self.feature_adapters[1](p4)  # 160 -> 80 channels
        p5_reduced = self.feature_adapters[2](p5)  # 448 -> 224 channels
        
        # Forward through optimized neck and head
        neck_output = self.neck(p3_reduced, p4_reduced, p5_reduced)
        return self.head(neck_output)