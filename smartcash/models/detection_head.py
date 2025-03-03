# File: models/detection_head.py 
# Author: Alfrida Sabar
# Deskripsi: Implementasi detection head dengan opsi multi-layer

from typing import Dict, List, Optional
import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    """Detection head yang bisa menangani single atau multi layer"""
    
    # Konfigurasi layer yang didukung dengan nama kelas aktual
    LAYER_CONFIG = {
        'banknote': {
            'num_classes': 7,  # 7 denominasi ('001', '002', '005', '010', '020', '050', '100')
            'description': 'Deteksi uang kertas utuh'
        },
        'nominal': {
            'num_classes': 7,  # 7 area nominal ('l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100')
            'description': 'Deteksi area nominal'
        },
        'security': {
            'num_classes': 3,  # 3 fitur keamanan ('l3_sign', 'l3_text', 'l3_thread')
            'description': 'Deteksi fitur keamanan'
        }
    }
    
    def __init__(
        self,
        in_channels: List[int],
        layers: Optional[List[str]] = None,  # Layer yang akan diaktifkan
        num_anchors: int = 3
    ):
        super().__init__()
        
        # Jika layers tidak dispesifikasi, gunakan banknote saja
        self.active_layers = layers or ['banknote']
        
        # Validasi layer yang diminta
        for layer in self.active_layers:
            if layer not in self.LAYER_CONFIG:
                raise ValueError(f"Layer {layer} tidak didukung")
        
        # Buat head untuk setiap kombinasi layer dan skala
        self.heads = nn.ModuleDict()
        for layer in self.active_layers:
            heads_per_scale = nn.ModuleList()
            num_classes = self.LAYER_CONFIG[layer]['num_classes']
            
            for ch in in_channels:
                heads_per_scale.append(
                    self._build_single_head(ch, num_classes, num_anchors)
                )
                
            self.heads[layer] = heads_per_scale
            
    def _build_single_head(
        self,
        in_ch: int,
        num_classes: int,
        num_anchors: int
    ) -> nn.Sequential:
        """Buat detection head untuk satu skala"""
        return nn.Sequential(
            self._conv_block(in_ch, in_ch//2),
            self._conv_block(in_ch//2, in_ch),
            nn.Conv2d(
                in_ch,
                num_anchors * (5 + num_classes),
                kernel_size=1
            )
        )
        
    def _conv_block(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                padding=kernel_size//2,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass detection head
        
        Args:
            features: List feature map dari FPN
            
        Returns:
            Dict berisi prediksi untuk setiap layer
            Format: {
                'banknote': [pred_s, pred_m, pred_l],
                'nominal': [pred_s, pred_m, pred_l],
                ...
            }
        """
        predictions = {}
        
        for layer_name, heads in self.heads.items():
            layer_preds = []
            num_classes = self.LAYER_CONFIG[layer_name]['num_classes']
            
            for feat, head in zip(features, heads):
                bs, _, h, w = feat.shape
                pred = head(feat)
                
                # Reshape ke format YOLO
                pred = pred.view(bs, 3, 5 + num_classes, h, w)
                pred = pred.permute(0, 1, 3, 4, 2)
                layer_preds.append(pred)
                
            predictions[layer_name] = layer_preds
            
        return predictions