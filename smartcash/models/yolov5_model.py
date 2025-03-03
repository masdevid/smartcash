# File: models/yolov5_model.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi model YOLOv5 yang bisa menggunakan CSPDarknet atau EfficientNet sebagai backbone

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union

from smartcash.models.backbones.base import BaseBackbone
from smartcash.models.backbones.cspdarknet import CSPDarknet
from smartcash.models.backbones.efficientnet import EfficientNetBackbone
from smartcash.models.losses import YOLOLoss
from smartcash.utils.logger import SmartCashLogger

class YOLOv5Model(nn.Module):
    """
    Model YOLOv5 dengan backbone yang bisa diganti.
    Mendukung penggunaan CSPDarknet (default) atau EfficientNet.
    """
    
    def __init__(
        self,
        num_classes: int = 7,  # 7 denominasi Rupiah
        backbone_type: str = "cspdarknet",
        pretrained: bool = True,
        layers: Optional[List[str]] = None,  # Parameter layers, bukan detection_layers
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.num_classes = num_classes
        
        # Default layers jika None
        self.active_layers = layers or ['banknote']
        
        # Inisialisasi backbone
        self.backbone = self._create_backbone(backbone_type, pretrained)
        
        # Feature Pyramid Network
        self.fpn = self._build_fpn(self.backbone.get_output_channels())
        
        # Detection Head
        self.head = self._build_head(
            in_channels=[256, 512, 1024],  # Channel setelah FPN
            num_classes=num_classes,
            layers=self.active_layers  # Gunakan active_layers di sini
        )
        
        self.logger.info(
            f"✨ Model YOLOv5 siap dengan:\n"
            f"   Backbone: {backbone_type}\n"
            f"   Classes: {num_classes}\n"
            f"   Layers: {self.active_layers}\n"
            f"   Pretrained: {pretrained}"
        )
        
    def _create_backbone(
        self,
        backbone_type: str,
        pretrained: bool
    ) -> BaseBackbone:
        """Buat backbone sesuai tipe yang dipilih"""
        if backbone_type == "cspdarknet":
            return CSPDarknet(pretrained=pretrained)
        elif backbone_type == "efficientnet":
            return EfficientNetBackbone(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone_type} tidak didukung")
            
    def _build_fpn(self, in_channels: List[int]) -> nn.ModuleList:
        """
        Build Feature Pyramid Network
        
        Args:
            in_channels: Channel dari setiap stage backbone
        """
        fpn_layers = nn.ModuleList()
        
        # Bottom-up pathway (sesuai YOLOv5)
        for i, ch in enumerate(in_channels):
            fpn_layers.append(
                nn.Sequential(
                    self._conv(ch, in_channels[i]//2, 1),
                    self._conv(in_channels[i]//2, in_channels[i], 3),
                    self._conv(in_channels[i], in_channels[i]//2, 1)
                )
            )
            
        return fpn_layers
        
    def _build_head(
        self,
        in_channels: List[int],
        num_classes: int,
        layers: List[str] = ['banknote']  # Default ke banknote layer saja
    ) -> nn.ModuleDict:  # Return ModuleDict bukan ModuleList
        """Build detection head untuk setiap layer dan skala."""
        heads = nn.ModuleDict()  # Gunakan ModuleDict untuk menyimpan head per layer
        
        # Buat head untuk setiap layer
        for layer_name in layers:
            # Buat ModuleList untuk skala P3, P4, P5 per layer
            layer_heads = nn.ModuleList()
            
            # Tambahkan head untuk setiap skala (P3, P4, P5)
            for ch in in_channels:
                layer_heads.append(
                    nn.Sequential(
                        self._conv(ch, ch//2, 1),
                        self._conv(ch//2, ch, 3),
                        nn.Conv2d(
                            ch,
                            3 * (5 + num_classes),  # 3 anchor, 5 bbox params, num_classes
                            1
                        )
                    )
                )
            
            # Tambahkan head layer ke heads
            heads[layer_name] = layer_heads
            
        return heads
        
    def _conv(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Helper untuk membuat convolution block"""
        return nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride,
                kernel_size//2,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass model
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dict berisi prediksi untuk setiap layer dan skala
            Format: {
                'banknote': [pred_s, pred_m, pred_l],
                'nominal': [pred_s, pred_m, pred_l],
                ...
            }
        """
        # Backbone features
        features = self.backbone(x)
        
        # Feature Pyramid Network
        fpn_features = []
        for feat, fpn in zip(features, self.fpn):
            fpn_features.append(fpn(feat))
            
        # Dict untuk menyimpan hasil prediksi per layer
        predictions = {}
        
        # Deteksi untuk setiap layer
        for layer_name, layer_heads in self.head.items():
            layer_preds = []
            
            # Prediksi untuk setiap skala pada layer ini
            for feat, head in zip(fpn_features, layer_heads):
                # Deteksi dan reshape ke format YOLO
                bs, _, h, w = feat.shape
                pred = head(feat)
                
                # [B, anchors*(5+classes), H, W] -> [B, anchors, H, W, 5+classes]
                pred = pred.view(bs, 3, 5 + self.num_classes, h, w)
                pred = pred.permute(0, 1, 3, 4, 2)
                
                layer_preds.append(pred)
            
            # Simpan prediksi layer
            predictions[layer_name] = layer_preds
            
        return predictions
        
    def compute_loss(
        self,
        predictions: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss
        
        Args:
            predictions: Prediksi model (single atau multi layer)
            targets: Ground truth labels (single atau multi layer)
            
        Returns:
            Dict berisi komponen loss
        """
        if not hasattr(self, 'loss_fn'):
            # Inisialisasi loss function jika belum ada
            self.loss_fn = YOLOLoss(
                layer_weights={'banknote': 1.0} if len(self.head.active_layers) == 1
                else {layer: 1.0 for layer in self.head.active_layers}
            )
            
        return self.loss_fn(predictions, targets)