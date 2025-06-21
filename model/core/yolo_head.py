"""
File: smartcash/model/core/yolo_head.py
Deskripsi: YOLO detection head untuk currency detection dengan multi-layer support
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from smartcash.common.logger import get_logger

class YOLOHead(nn.Module):
    """🎯 YOLO detection head untuk currency detection"""
    
    def __init__(self, in_channels: List[int], detection_layers: List[str] = None,
                 layer_mode: str = 'single', num_classes: int = 7, img_size: int = 640,
                 num_anchors: int = 3):
        super().__init__()
        
        self.logger = get_logger("model.yolo_head")
        self.in_channels = in_channels
        self.detection_layers = detection_layers or ['banknote']
        self.layer_mode = layer_mode
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_anchors = num_anchors
        
        # Layer configuration untuk currency detection
        self.layer_config = {
            'banknote': {'num_classes': 7, 'description': 'Main banknote detection'},
            'nominal': {'num_classes': 7, 'description': 'Nominal value detection'},
            'security': {'num_classes': 3, 'description': 'Security features detection'}
        }
        
        # Build detection heads
        self._build_detection_heads()
        
        self.logger.info(f"🎯 YOLO Head initialized | Mode: {layer_mode} | Layers: {detection_layers}")
    
    def _build_detection_heads(self):
        """🔧 Build detection heads berdasarkan layer mode"""
        
        if self.layer_mode == 'single':
            # Single layer mode: satu head untuk primary layer
            primary_layer = self.detection_layers[0]
            layer_classes = self.layer_config[primary_layer]['num_classes']
            
            self.heads = nn.ModuleDict({
                primary_layer: self._create_layer_heads(layer_classes)
            })
            
        elif self.layer_mode == 'multilayer':
            # Multi-layer mode: separate heads untuk setiap layer
            self.heads = nn.ModuleDict()
            
            for layer_name in self.detection_layers:
                if layer_name in self.layer_config:
                    layer_classes = self.layer_config[layer_name]['num_classes']
                    self.heads[layer_name] = self._create_layer_heads(layer_classes)
        
        self.logger.debug(f"🔧 Built {len(self.heads)} detection heads")
    
    def _create_layer_heads(self, num_classes: int) -> nn.ModuleList:
        """🎯 Create detection heads untuk different scales (P3, P4, P5)"""
        
        heads = nn.ModuleList()
        
        for in_ch in self.in_channels:
            # YOLO head: conv layers + final prediction layer
            head = nn.Sequential(
                self._conv_block(in_ch, in_ch * 2, 3, 1, 1),
                self._conv_block(in_ch * 2, in_ch, 1, 1, 0),
                self._conv_block(in_ch, in_ch * 2, 3, 1, 1),
                nn.Conv2d(in_ch * 2, self.num_anchors * (5 + num_classes), 1)
            )
            heads.append(head)
        
        return heads
    
    def _conv_block(self, in_ch: int, out_ch: int, k: int, s: int, p: int) -> nn.Sequential:
        """🔧 Convolution block dengan BatchNorm dan activation"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """🔄 Forward pass untuk semua detection layers"""
        
        if len(features) != len(self.in_channels):
            raise ValueError(f"❌ Expected {len(self.in_channels)} features, got {len(features)}")
        
        outputs = {}
        
        for layer_name, layer_heads in self.heads.items():
            layer_outputs = []
            
            for i, (feature, head) in enumerate(zip(features, layer_heads)):
                # Raw prediction
                pred = head(feature)
                
                # Reshape untuk YOLO format
                batch_size, _, height, width = pred.shape
                pred = pred.view(
                    batch_size, 
                    self.num_anchors, 
                    5 + self.layer_config[layer_name]['num_classes'],
                    height, 
                    width
                ).permute(0, 1, 3, 4, 2).contiguous()
                
                layer_outputs.append(pred)
            
            outputs[layer_name] = layer_outputs
        
        return outputs
    
    def get_output_channels(self) -> List[int]:
        """📊 Get output channels untuk setiap scale"""
        return self.in_channels
    
    def get_head_info(self) -> Dict[str, Any]:
        """ℹ️ Get information tentang detection head"""
        return {
            'layer_mode': self.layer_mode,
            'detection_layers': self.detection_layers,
            'layer_config': self.layer_config,
            'num_classes_per_layer': {
                layer: self.layer_config[layer]['num_classes'] 
                for layer in self.detection_layers
            },
            'total_heads': len(self.heads),
            'scales': len(self.in_channels),
            'img_size': self.img_size,
            'num_anchors': self.num_anchors
        }