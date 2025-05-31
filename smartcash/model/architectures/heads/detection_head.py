"""
File: smartcash/model/architectures/heads/detection_head.py
Deskripsi: Detection Head implementation for YOLOv5
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import HeadError
from smartcash.model.config.model_constants import LAYER_CONFIG, DETECTION_LAYERS
from smartcash.model.utils.layer_validator import validate_layer_params

class DetectionHead(nn.Module):
    """YOLOv5 Detection Head untuk deteksi objek multi-layer dengan berbagai skala."""
    
    def __init__(self, in_channels: List[int], detection_layers: List[str] = None, 
                 num_classes: int = None, layer_mode: str = 'single', img_size: int = 640,
                 use_attention: bool = False, logger: Optional[SmartCashLogger] = None):
        """Inisialisasi Detection Head dengan konfigurasi channels dan layers."""
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.in_channels = in_channels
        self.layer_mode = layer_mode
        self.img_size = img_size
        self.use_attention = use_attention
        
        # Jika detection_layers tidak diberikan, gunakan default ['banknote']
        if detection_layers is None:
            detection_layers = ['banknote']
            
        # Log parameter awal sebelum validasi
        self.logger.info(f"📝 Parameter awal: layer_mode={layer_mode}, detection_layers={detection_layers}")
        
        # Gunakan fungsi validate_layer_params untuk validasi
        # PENTING: Jika mode multilayer dengan multiple layers, JANGAN ubah ke single
        if layer_mode == 'multilayer' and len([l for l in detection_layers if l in DETECTION_LAYERS]) >= 2:
            # Jika sudah multilayer dengan 2+ layer valid, pertahankan mode ini
            valid_layer_mode = 'multilayer'
            valid_detection_layers = [l for l in detection_layers if l in DETECTION_LAYERS]
            if len(valid_detection_layers) != len(detection_layers):
                self.logger.warning(f"⚠️ Beberapa detection_layers tidak valid, hanya menggunakan yang valid: {valid_detection_layers}")
        else:
            # Gunakan validasi standar untuk kasus lainnya
            valid_layer_mode, valid_detection_layers = validate_layer_params(layer_mode, detection_layers)
            
        # Log hasil validasi
        self.logger.info(f"📝 Setelah validasi: layer_mode={valid_layer_mode}, detection_layers={valid_detection_layers}")
        
        # Update atribut dengan nilai yang sudah divalidasi
        self.layer_mode = valid_layer_mode
        self.detection_layers = valid_detection_layers
        
        # Untuk mode single dengan banyak layer, beri peringatan
        if self.layer_mode == 'single' and len(self.detection_layers) > 1:
            self.logger.warning(f"⚠️ Mode single dengan {len(self.detection_layers)} detection_layers, hanya layer pertama yang akan digunakan secara efektif.")
        
        self.num_classes = num_classes
        
        # Validasi detection layers dan buat head untuk setiap kombinasi layer dan skala
        for layer in self.detection_layers: 
            if layer not in LAYER_CONFIG: 
                raise HeadError(f"❌ Layer {layer} tidak didukung. Layer yang didukung: {list(LAYER_CONFIG.keys())}")
        
        # Hitung total kelas berdasarkan layer_mode
        if self.layer_mode == 'multilayer' and self.num_classes is None:
            # Dalam mode multilayer, jumlah kelas adalah total dari semua layer
            self.total_classes = sum(LAYER_CONFIG[layer]['num_classes'] for layer in self.detection_layers)
            self.logger.info(f"🔢 Mode multilayer: total kelas = {self.total_classes}")
        else:
            # Dalam mode single, gunakan num_classes dari parameter atau dari layer pertama
            self.total_classes = self.num_classes if self.num_classes is not None else LAYER_CONFIG[self.detection_layers[0]]['num_classes']
        
        # Buat head untuk setiap kombinasi layer dan skala
        if self.layer_mode == 'single':
            # Mode single: buat head terpisah untuk setiap layer
            self.heads = nn.ModuleDict({
                layer: nn.ModuleList([self._build_single_head(
                    ch, 
                    self.num_classes if self.num_classes is not None else LAYER_CONFIG[layer]['num_classes'], 
                    3
                ) for ch in in_channels])
                for layer in self.detection_layers
            })
        else:
            # Mode multilayer: buat satu head dengan total kelas dari semua layer
            self.heads = nn.ModuleDict({
                'multilayer': nn.ModuleList([self._build_single_head(
                    ch, self.total_classes, 3
                ) for ch in in_channels])
            })
            
            # Simpan offset kelas untuk setiap layer dalam mode multilayer
            self.class_offsets = {}
            offset = 0
            for layer in self.detection_layers:
                layer_classes = LAYER_CONFIG[layer]['num_classes']
                self.class_offsets[layer] = (offset, offset + layer_classes)
                offset += layer_classes
        
        self.logger.info(f"✨ Detection Head diinisialisasi:\n   • Layer mode: {self.layer_mode}\n   • Detection layers: {self.detection_layers}\n   • Input channels: {in_channels}\n   • Total kelas: {self.total_classes}\n   • Total heads: {len(self.heads)}\n   • Image size: {self.img_size}\n   • Use attention: {self.use_attention}")
            
    def _build_single_head(self, in_ch: int, num_classes: int, num_anchors: int) -> nn.Sequential:
        """Buat detection head untuk satu skala."""
        return nn.Sequential(
            self._conv_block(in_ch, in_ch//2),
            self._conv_block(in_ch//2, in_ch),
            nn.Conv2d(in_ch, num_anchors * (5 + num_classes), kernel_size=1)
        )
        
    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
        """Buat convolution block dengan batch normalization dan aktivasi."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """Forward pass dengan input features dari backbone+neck [P3, P4, P5]."""
        try:
            # Validasi input
            if len(features) != len(self.in_channels): 
                raise HeadError(f"❌ Jumlah feature maps ({len(features)}) tidak sesuai dengan jumlah in_channels ({len(self.in_channels)})")
            
            # Mode multilayer: gunakan satu head untuk semua layer
            if self.layer_mode == 'multilayer':
                results = {layer: [] for layer in self.detection_layers}
                
                for i, feat in enumerate(features):
                    # Ambil head multilayer dan proses feature
                    head = self.heads['multilayer'][i]
                    bs, _, h, w = feat.shape
                    
                    # Reshape ke format YOLO [B, anchors*(5+total_classes), H, W] -> [B, anchors, H, W, 5+total_classes]
                    pred = head(feat).view(bs, 3, 5 + self.total_classes, h, w).permute(0, 1, 3, 4, 2)
                    
                    # Distribusikan prediksi ke layer yang sesuai berdasarkan offset kelas
                    for layer_name in self.detection_layers:
                        start_idx, end_idx = self.class_offsets[layer_name]
                        layer_pred = pred.clone()
                        
                        # Ambil hanya kelas yang relevan untuk layer ini
                        # Format: [x, y, w, h, conf, classes...]
                        # Simpan koordinat box dan confidence (indeks 0-4)
                        # Ambil kelas yang relevan (indeks 5+start_idx hingga 5+end_idx)
                        layer_classes = end_idx - start_idx
                        layer_pred_classes = layer_pred[:, :, :, :, 5:].clone()
                        
                        # Buat tensor baru dengan ukuran yang sesuai untuk layer ini
                        new_pred = torch.zeros(bs, 3, h, w, 5 + layer_classes, device=pred.device)
                        new_pred[:, :, :, :, :5] = layer_pred[:, :, :, :, :5]  # Salin koordinat dan confidence
                        new_pred[:, :, :, :, 5:] = layer_pred_classes[:, :, :, :, start_idx:end_idx]  # Salin kelas yang relevan
                        
                        results[layer_name].append(new_pred)
            else:
                # Mode single: proses setiap layer terpisah
                results = {layer: [] for layer in self.detection_layers}
                head_idx = 0
                
                for layer_name in self.detection_layers:
                    for feat in features:
                        # Ambil head, proses feature, dan reshape output
                        head, head_idx = self.heads[layer_name][head_idx % len(features)], head_idx + 1
                        bs, _, h, w = feat.shape
                        num_classes = self.num_classes if self.num_classes is not None else LAYER_CONFIG[layer_name]['num_classes']
                        
                        # Reshape ke format YOLO [B, anchors*(5+classes), H, W] -> [B, anchors, H, W, 5+classes]
                        pred = head(feat).view(bs, 3, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2)
                        results[layer_name].append(pred)
            
            return results
        except Exception as e:
            self.logger.error(f"❌ Forward pass gagal: {str(e)}")
            raise HeadError(f"Forward pass gagal: {str(e)}")
    
    def get_config(self) -> Dict:
        """Dapatkan konfigurasi detection head."""
        config = {
            'layer_mode': self.layer_mode,
            'layers': self.detection_layers,
            'layer_config': {layer: LAYER_CONFIG[layer] for layer in self.detection_layers},
            'total_classes': self.total_classes,
            'img_size': self.img_size,
            'use_attention': self.use_attention
        }
        
        if self.layer_mode == 'multilayer':
            config['class_offsets'] = self.class_offsets
            
        return config