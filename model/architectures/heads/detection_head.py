"""
File: smartcash/model/architectures/heads/detection_head.py
Deskripsi: Detection head implementation for YOLOv5 dengan ekspor LAYER_CONFIG
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import HeadError

# Konfigurasi layer yang didukung dengan nama kelas aktual
# Ekspor ini sebagai variabel modul agar bisa diimpor langsung dari file lain
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

class DetectionHead(nn.Module):
    """
    Detection Head untuk YOLOv5 dengan dukungan multi-layer.
    
    Mendukung deteksi beberapa layer berbeda secara simultan, seperti:
    - Layer banknote untuk deteksi uang kertas utuh
    - Layer nominal untuk deteksi area nominal
    - Layer security untuk deteksi fitur keamanan
    """
    
    def __init__(
        self,
        in_channels: List[int],
        layers: Optional[List[str]] = None,  # Layer yang akan diaktifkan
        num_anchors: int = 3,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi Detection Head.
        
        Args:
            in_channels: List jumlah channel untuk setiap feature map input (dari neck)
            layers: List nama layer yang akan diaktifkan (default: ['banknote'])
            num_anchors: Jumlah anchor per cell
            logger: Logger untuk mencatat proses (opsional)
            
        Raises:
            HeadError: Jika parameter tidak valid atau layer tidak didukung
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        # Jika layers tidak dispesifikasi, gunakan banknote saja
        self.layers = layers or ['banknote']
        
        # Validasi layer yang diminta
        for layer in self.layers:
            if layer not in LAYER_CONFIG:
                supported = list(LAYER_CONFIG.keys())
                raise HeadError(f"❌ Layer '{layer}' tidak didukung. Layer yang didukung: {supported}")
        
        # Buat head untuk setiap kombinasi layer dan skala
        self.heads = nn.ModuleDict()
        for layer in self.layers:
            heads_per_scale = nn.ModuleList()
            num_classes = LAYER_CONFIG[layer]['num_classes']
            
            for ch in in_channels:
                heads_per_scale.append(
                    self._build_single_head(ch, num_classes, num_anchors)
                )
                
            self.heads[layer] = heads_per_scale
            
        self.logger.info(
            f"✅ DetectionHead diinisialisasi:\n"
            f"   • Layers aktif: {self.layers}\n"
            f"   • Input channels: {in_channels}\n"
            f"   • Anchors per cell: {num_anchors}"
        )
            
    def _build_single_head(
        self,
        in_ch: int,
        num_classes: int,
        num_anchors: int
    ) -> nn.Sequential:
        """
        Buat detection head untuk satu skala.
        
        Args:
            in_ch: Jumlah channel input
            num_classes: Jumlah kelas untuk deteksi
            num_anchors: Jumlah anchor per cell
            
        Returns:
            nn.Sequential: Layer detection head
        """
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
        """
        Buat convolution block dengan batch normalization dan aktivasi.
        
        Args:
            in_ch: Jumlah channel input
            out_ch: Jumlah channel output
            kernel_size: Ukuran kernel
            
        Returns:
            nn.Sequential: Convolution block
        """
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
        Forward pass detection head.
        
        Args:
            features: List feature map dari neck
            
        Returns:
            Dict berisi prediksi untuk setiap layer
            Format: {
                'banknote': [pred_s, pred_m, pred_l],
                'nominal': [pred_s, pred_m, pred_l],
                ...
            }
            
        Raises:
            HeadError: Jika forward pass gagal
        """
        try:
            # Validasi input features
            if not features or len(features) != len(next(iter(self.heads.values()))):
                raise HeadError(
                    f"❌ Jumlah feature maps ({len(features) if features else 0}) "
                    f"tidak sesuai dengan jumlah scales yang diharapkan "
                    f"({len(next(iter(self.heads.values())))}"
                )
            
            predictions = {}
            
            for layer_name, heads in self.heads.items():
                layer_preds = []
                num_classes = LAYER_CONFIG[layer_name]['num_classes']
                
                # Detection head untuk setiap skala
                for feat, head in zip(features, heads):
                    bs, _, h, w = feat.shape
                    pred = head(feat)
                    
                    # Reshape ke format YOLO
                    # [B, anchors*(5+classes), H, W] -> [B, anchors, H, W, 5+classes]
                    pred = pred.view(bs, 3, 5 + num_classes, h, w)
                    pred = pred.permute(0, 1, 3, 4, 2)
                    
                    layer_preds.append(pred)
                    
                predictions[layer_name] = layer_preds
                
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ DetectionHead forward pass gagal: {str(e)}")
            raise HeadError(f"Forward pass gagal: {str(e)}")
    
    def get_config(self) -> Dict:
        """
        Dapatkan konfigurasi detection head.
        
        Returns:
            Dict: Konfigurasi detection head
        """
        return {
            'layers': self.layers,
            'layer_config': {
                layer: LAYER_CONFIG[layer]
                for layer in self.layers
            }
        }