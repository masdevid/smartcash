# File: src/models/neck_head.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi FPN dan detection head dengan optimasi untuk deteksi nominal uang

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import ColoredLogger

class AdaptiveFPN(nn.Module):
    def __init__(self, channels, out_channels=256):
        super().__init__()
        self.logger = ColoredLogger('FPN')
        
        # Lateral connections with channel adaptation
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for c in channels[:-1]
        ])
        
        # Top-down pathway
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(len(channels)-1)
        ])
        
        self.logger.info("🔗 FPN initialized")

    def forward(self, features):
        results = []
        last_feature = features[-1]
        
        # Top-down path
        for lat, up, feat in zip(
            self.laterals[::-1], 
            self.upsamples[::-1], 
            features[-2::-1]
        ):
            # Process current level
            curr_lat = lat(feat)
            last_feature = up(last_feature)
            
            # Feature fusion
            last_feature = curr_lat + last_feature
            results.append(last_feature)
        
        return results[::-1] + [features[-1]]

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=7, num_anchors=3):
        super().__init__()
        self.logger = ColoredLogger('Head')
        self.nc = num_classes
        self.na = num_anchors
        self.no = num_classes + 5  # outputs per anchor
        
        # Optimized prediction heads
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
                # Depth-wise separable convolution for efficiency
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels, self.no * self.na, 1),
            ) for _ in range(3)  # 3 detection levels
        ])
        
        self.logger.info("🎯 Detection head initialized")

    def forward(self, features):
        results = []
        
        for feat, head in zip(features, self.pred_heads):
            # Process each feature level
            pred = head(feat)
            bs, _, h, w = pred.shape
            
            # Reshape predictions
            pred = pred.view(bs, self.na, self.no, h, w)
            pred = pred.permute(0, 1, 3, 4, 2)  # (bs, na, h, w, no)
            
            results.append(pred)
        
        return results

class NeckHead(nn.Module):
    def __init__(self, channels, num_classes=7):
        super().__init__()
        self.fpn = AdaptiveFPN(channels)
        self.head = DetectionHead(channels[-1], num_classes)
        
    def forward(self, features):
        fpn_features = self.fpn(features)
        return self.head(fpn_features)