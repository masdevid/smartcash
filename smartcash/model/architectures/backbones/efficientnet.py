"""
File: smartcash/model/architectures/backbones/efficientnet.py
Deskripsi: EfficientNet backbone implementation for YOLOv5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import timm

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import BackboneError, UnsupportedBackboneError
from smartcash.model.architectures.backbones.base import BaseBackbone

from smartcash.model.config.model_constants import (
    SUPPORTED_EFFICIENTNET_MODELS, 
    EFFICIENTNET_CHANNELS, 
    YOLO_CHANNELS,
    DEFAULT_EFFICIENTNET_INDICES
)

class FeatureAdapter(nn.Module):
    """Adapter untuk memetakan feature maps dari EfficientNet ke format YOLOv5."""
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        """Inisialisasi adapter dengan channel adaptation dan opsional attention."""
        super().__init__()
        self.channel_adapt = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.attention = ChannelAttention(out_channels) if use_attention else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass dengan channel adaptation dan opsional attention."""
        x = self.channel_adapt(x)
        return self.attention(x) * x if self.attention is not None else x

class ChannelAttention(nn.Module):
    """Channel Attention untuk memperkuat feature penting dengan dual pooling."""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet backbone untuk YOLOv5 dengan ekstraksi feature maps dan adaptasi channel."""
    
    def __init__(self, model_name: str = 'efficientnet_b4', pretrained: bool = True, feature_indices: Optional[List[int]] = None, out_channels: Optional[List[int]] = None, use_attention: bool = True, testing_mode: bool = False, logger: Optional[SmartCashLogger] = None):
        """Inisialisasi EfficientNet backbone dengan konfigurasi model."""
        super().__init__(logger=logger)
        self.model_name = model_name
        self.testing_mode = testing_mode
        self.feature_indices = feature_indices or DEFAULT_EFFICIENTNET_INDICES
        self.use_attention = use_attention
        
        # Validasi model
        if model_name not in SUPPORTED_EFFICIENTNET_MODELS:
            raise BackboneError(f"❌ Model {model_name} tidak didukung. Model yang didukung: {', '.join(SUPPORTED_EFFICIENTNET_MODELS)}")
        
        # Setup model berdasarkan mode
        if testing_mode:
            self.logger.info(f"🧪 Membuat dummy EfficientNet backbone untuk testing: {model_name}")
            self._setup_dummy_model_for_testing()
        else:
            try:
                self.logger.info(f"🔄 Loading EfficientNet backbone: {model_name}")
                self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=self.feature_indices)
                
                # Validasi channels dengan dummy input
                expected = EFFICIENTNET_CHANNELS[model_name]
                with torch.no_grad():
                    features = self.model(torch.zeros(1, 3, 224, 224))  # Gunakan dummy input untuk validasi
                    actual_channels = [f.shape[1] for f in features]
                    
                    if actual_channels != expected:
                        self.logger.warning(f"⚠️ Channels tidak sesuai ekspektasi: {actual_channels} vs {expected}")
                    
                    self.channels = actual_channels
                    self.logger.info(f"✅ EfficientNet backbone berhasil dimuat dengan channels: {self.channels}")
            except Exception as e:
                raise BackboneError(f"❌ Gagal memuat EfficientNet backbone: {str(e)}")
            
    def _setup_dummy_model_for_testing(self):
        """Membuat model dummy untuk testing tanpa memerlukan pretrained weights."""
        # Langsung gunakan YOLO_CHANNELS untuk output channels
        self.out_channels = YOLO_CHANNELS
        
        # Simpan juga channels asli EfficientNet untuk referensi
        self.efficientnet_channels = EFFICIENTNET_CHANNELS[self.model_name]
        
        # Untuk mode testing, kita langsung menggunakan YOLO_CHANNELS sebagai output
        self.channels = self.out_channels
        
        # Setup feature indices
        self.feature_indices = [0, 1, 2]  # Simplified indices for dummy model
        
        # Buat layer dummy yang langsung menghasilkan channel sesuai dengan YOLO_CHANNELS
        self.dummy_layers = nn.ModuleList()
        in_channels = 3
        
        # Buat layer dummy untuk setiap output channel yang diharapkan
        for i, out_ch in enumerate(self.out_channels):
            # Buat layer konvolusi sederhana yang langsung menghasilkan channel yang diharapkan
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.dummy_layers.append(layer)
            in_channels = out_ch
        
        self.logger.info(f"✅ Dummy model untuk EfficientNet-{self.model_name} berhasil dibuat dengan {len(self.out_channels)} feature maps")
        self.logger.info(f"   • Output channels: {self.out_channels} (sesuai dengan YOLO_CHANNELS)")
        
        # Override metode forward untuk menggunakan model dummy
        self.forward = self._forward_dummy
        
    def _forward_dummy(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass dengan model dummy untuk testing."""
        features = []
        
        # Buat dummy feature maps dengan channel yang sesuai dengan YOLO_CHANNELS
        for i, out_ch in enumerate(self.out_channels):
            # Downsample input sesuai dengan level feature map
            # P3 = 1/8, P4 = 1/16, P5 = 1/32 dari input asli
            scale_factor = 1 / (2 ** (i + 3))
            h, w = x.shape[2], x.shape[3]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Buat dummy tensor dengan channel yang sesuai
            batch_size = x.shape[0]
            dummy_feature = torch.zeros(batch_size, out_ch, new_h, new_w, device=x.device)
            
            # Isi dengan nilai random untuk simulasi feature map
            dummy_feature.normal_(0, 0.02)
            
            features.append(dummy_feature)
            
            # Log untuk debugging
            self.logger.debug(f"🔍 Feature map {i}: shape={dummy_feature.shape}, channels={dummy_feature.shape[1]}")
        
        # Pastikan jumlah feature maps sesuai dengan yang diharapkan
        if len(features) != len(self.out_channels):
            self.logger.warning(f"⚠️ Jumlah feature maps ({len(features)}) tidak sesuai dengan yang diharapkan ({len(self.out_channels)})")
        
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass untuk EfficientNet backbone, mengembalikan feature maps."""
        if hasattr(self, 'model'):
            # Mode normal, gunakan model timm
            features = self.model(x)
            return features
        else:
            # Mode testing, gunakan _forward_dummy
            return self._forward_dummy(x)

    def get_output_channels(self) -> List[int]: 
        if hasattr(self, 'out_channels'):
            return self.out_channels
        else:
            self.out_channels = YOLO_CHANNELS
            return self.out_channels
    
    def get_output_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> List[Tuple[int, int]]:
        """Dapatkan dimensi spasial dari output feature maps untuk input_size tertentu."""
        width, height = input_size
        return [(height // 8, width // 8), (height // 16, width // 16), (height // 32, width // 32)]  # P3, P4, P5
        
    def get_info(self) -> Dict:
        """Dapatkan informasi backbone dalam bentuk dictionary.
        
        Returns:
            Dict: Informasi tentang backbone
        """
        return {
            'type': 'EfficientNet',
            'variant': self.model_name,
            'out_channels': self.out_channels,
            'feature_indices': self.feature_indices,
            'input_channels': self.actual_channels,
            'pretrained': True,  # Berdasarkan inisialisasi
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass dengan ekstraksi fitur dan adaptasi channel untuk input tensor."""
        try:
            features = self.model(x)  # Extract multi-scale features
            if len(features) != len(self.adapters): self.logger.warning(f"⚠️ Jumlah feature map ({len(features)}) tidak sesuai dengan jumlah adapters ({len(self.adapters)})!")
            adapted_features = [adapter(feat) for feat, adapter in zip(features, self.adapters)]  # Apply adapters
            self.validate_output(adapted_features, self.out_channels)  # Validasi output
            return adapted_features
        except Exception as e:
            self.logger.error(f"❌ Forward pass gagal: {str(e)}")
            raise BackboneError(f"Forward pass gagal: {str(e)}")