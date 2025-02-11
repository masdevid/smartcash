# File: src/models/efficientnet_backbone.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi backbone EfficientNet-B4 dengan optimasi memori dan fitur untuk SmartCash Detector

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from utils.logging import ColoredLogger

class CompoundScaler:
    def __init__(self, base_width=256, base_depth=3):
        self.base_w = base_width
        self.base_d = base_depth
        self.logger = ColoredLogger('Scaler')
        
    def scale(self, phi):
        # Optimized compound scaling
        alpha, beta, gamma = 1.2, 1.1, 1.15
        w = int(self.base_w * (alpha ** phi))
        d = max(3, int(self.base_d * (beta ** phi)))
        self.logger.info(f"📏 Scaled dims: w={w}, d={d}")
        return w, d

class FeatureProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv1x1(x)))

class EfficientNetBackbone(nn.Module):
    def __init__(self, phi=4):
        super().__init__()
        self.logger = ColoredLogger('Backbone')
        self.scaler = CompoundScaler()
        self.width, self.depth = self.scaler.scale(phi)
        
        # Load pretrained backbone
        self.logger.info("🔄 Loading EfficientNet-B4...")
        efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Extract stages with memory optimization
        stages = list(efficientnet.children())[0]
        self.stage1 = nn.Sequential(*stages[:2])
        self.stage2 = nn.Sequential(*stages[2:3])
        self.stage3 = nn.Sequential(*stages[3:5])
        self.stage4 = nn.Sequential(*stages[5:7])
        self.stage5 = nn.Sequential(*stages[7:])
        
        # Feature processors for each stage
        channels = self._get_channels()
        self.processors = nn.ModuleList([
            FeatureProcessor(c, self.width) 
            for c in channels
        ])
        
        self.logger.info("✅ Backbone initialized")
        
    def _get_channels(self):
        """Get output channels for each stage"""
        return [
            self.stage1[-1].out_channels,
            self.stage2[-1].out_channels,
            self.stage3[-1].out_channels,
            self.stage4[-1].out_channels,
            self.stage5[-1].out_channels
        ]

    def forward(self, x):
        features = []
        
        # Memory-efficient feature extraction
        feat = self.stage1(x)
        features.append(self.processors[0](feat))
        
        feat = self.stage2(feat)
        features.append(self.processors[1](feat))
        
        feat = self.stage3(feat)
        features.append(self.processors[2](feat))
        
        feat = self.stage4(feat)
        features.append(self.processors[3](feat))
        
        feat = self.stage5(feat)
        features.append(self.processors[4](feat))
        
        return features
    
class SmartCashYOLODetector(nn.Module):
    def __init__(self, nc=7, phi=4):
        super().__init__()
        self.backbone = EfficientNetBackbone(phi=phi)
        self.nc = nc
        self.no = nc + 5  # outputs per anchor
        
        # Detection heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone.channels[-1], 
                         self.no * 3,  # 3 anchors
                         kernel_size=1),
                nn.BatchNorm2d(self.no * 3)
            ) for _ in range(3)  # 3 detection levels
        ])

    def forward(self, x):
        # Get multi-scale features
        features = self.backbone(x)
        
        # Apply detection heads
        return [head(feat) for head, feat in zip(self.heads, features)]