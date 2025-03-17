"""
File: smartcash/model/architectures/backbones/efficientnet.py
Deskripsi: EfficientNet backbone implementation for YOLOv5
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
import timm

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import BackboneError
from smartcash.model.architectures.backbones.base import BaseBackbone

class FeatureAdapter(nn.Module):
    """
    Adapter khusus untuk memetakan feature maps dari EfficientNet ke format YOLOv5.
    Menyediakan adaptasi spasial dan adaptasi channel dengan 1x1 Conv + SiLU.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        """
        Inisialisasi Feature Adapter.
        
        Args:
            in_channels: Jumlah input channels
            out_channels: Jumlah output channels
            use_attention: Gunakan channel attention
        """
        super().__init__()
        
        # 1x1 Conv untuk channel adaptation
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Opsional: Channel Attention
        self.attention = None
        if use_attention:
            self.attention = ChannelAttention(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass untuk mengadaptasi feature."""
        # Channel adaptation
        x = self.channel_adapt(x)
        
        # Apply channel attention jika aktif
        if self.attention is not None:
            x = self.attention(x) * x
            
        return x

class ChannelAttention(nn.Module):
    """Modul Channel Attention untuk memperkuat feature penting."""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass attention."""
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out

class EfficientNetBackbone(BaseBackbone):
    """
    EfficientNet backbone untuk arsitektur YOLOv5 dengan adaptasi channel yang optimal.
    
    Menggunakan pretrained EfficientNet dari library timm dengan
    adaptasi channel output untuk kompatibilitas dengan arsitektur YOLOv5.
    """
    
    # Channel yang diharapkan dari berbagai varian EfficientNet
    EXPECTED_CHANNELS = {
        'efficientnet_b0': [24, 48, 208],  # P3, P4, P5 stages
        'efficientnet_b1': [32, 88, 320],
        'efficientnet_b2': [32, 112, 352], 
        'efficientnet_b3': [40, 112, 384],
        'efficientnet_b4': [56, 160, 448],
        'efficientnet_b5': [64, 176, 512],
    }
    
    # Output channels standar yang digunakan YOLOv5
    YOLO_CHANNELS = [128, 256, 512]
    
    def __init__(
        self, 
        pretrained: bool = True, 
        model_name: str = 'efficientnet_b4',
        out_indices: Tuple[int, ...] = (2, 3, 4),  # P3, P4, P5 stages
        use_attention: bool = True,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi EfficientNet backbone.
        
        Args:
            pretrained: Gunakan pretrained weights atau tidak
            model_name: Nama model EfficientNet (efficientnet_b0 hingga efficientnet_b7)
            out_indices: Indeks untuk output feature map
            use_attention: Gunakan channel attention untuk adaptasi
            logger: Logger untuk mencatat proses (opsional)
        
        Raises:
            BackboneError: Jika model_name tidak didukung
        """
        super().__init__(logger=logger)
        self.model_name = model_name
        
        try:
            # Validasi model name
            if model_name not in self.EXPECTED_CHANNELS:
                supported = list(self.EXPECTED_CHANNELS.keys())
                raise BackboneError(
                    f"❌ Model {model_name} tidak didukung. "
                    f"Model yang didukung: {supported}"
                )
            
            # Load pretrained EfficientNet
            self.logger.info(f"🔄 Loading EfficientNet backbone: {model_name}")
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )
            
            # Deteksi output channel secara dinamis
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 640, 640)
                outputs = self.model(dummy_input)
                actual_channels = [o.shape[1] for o in outputs]
                
                # Log informasi channel
                expected = self.EXPECTED_CHANNELS[model_name]
                self.logger.debug(f"🔍 {model_name} channels (expected): {expected}")
                self.logger.debug(f"🔍 {model_name} channels (actual): {actual_channels}")
                
                # Verifikasi channel sesuai dengan ekspektasi
                if actual_channels != expected:
                    self.logger.warning(
                        f"⚠️ Channel yang diharapkan ({expected}) tidak sesuai dengan "
                        f"channel sebenarnya ({actual_channels})! Akan mengadaptasi sesuai output aktual."
                    )
                
                # Simpan channel aktual untuk referensi
                self.actual_channels = actual_channels
            
            # Buat adapter layer untuk konversi channel + attention
            self.adapters = nn.ModuleList([
                FeatureAdapter(in_ch, out_ch, use_attention=use_attention)
                for in_ch, out_ch in zip(actual_channels, self.YOLO_CHANNELS)
            ])
            
            self.logger.success(
                f"✅ Berhasil load EfficientNet backbone:\n"
                f"   • Model: {model_name}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Input channels: {actual_channels}\n"
                f"   • Output channels: {self.YOLO_CHANNELS}\n"
                f"   • Attention: {'Aktif' if use_attention else 'Nonaktif'}"
            )
        except BackboneError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"❌ Gagal load {model_name}: {str(e)}")
            raise BackboneError(f"Gagal load {model_name}: {str(e)}")
    
    def get_output_channels(self) -> List[int]:
        """
        Dapatkan jumlah output channel untuk setiap level feature.
        
        Returns:
            List[int]: Jumlah channel dari setiap feature map yang akan diteruskan ke neck
        """
        return self.YOLO_CHANNELS
    
    def get_output_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> List[Tuple[int, int]]:
        """
        Dapatkan dimensi spasial dari output feature maps.
        
        Args:
            input_size: Ukuran input (width, height)
            
        Returns:
            List[Tuple[int, int]]: Ukuran spasial untuk setiap output feature map
        """
        # Untuk input 640x640, output stride biasanya adalah 8, 16, 32
        width, height = input_size
        return [
            (height // 8, width // 8),   # P3
            (height // 16, width // 16), # P4
            (height // 32, width // 32)  # P5
        ]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass dengan ekstraksi fitur dan adaptasi channel.
        
        Args:
            x: Input tensor dengan shape [batch_size, channels, height, width]
            
        Returns:
            List[torch.Tensor]: Feature maps dengan channel yang sudah diadaptasi
            
        Raises:
            BackboneError: Jika forward pass gagal
        """
        try:
            # Extract multi-scale features dari EfficientNet
            features = self.model(x)
            
            # Verifikasi output shape sesuai ekspektasi
            if len(features) != len(self.adapters):
                self.logger.warning(
                    f"⚠️ Jumlah feature map ({len(features)}) tidak sesuai dengan "
                    f"jumlah adapters ({len(self.adapters)})!"
                )
            
            # Apply channel adapters dengan attention
            adapted_features = []
            for feat, adapter in zip(features, self.adapters):
                adapted = adapter(feat)
                adapted_features.append(adapted)
            
            # Validasi output feature
            self.validate_output(adapted_features, self.YOLO_CHANNELS)
                
            return adapted_features
            
        except Exception as e:
            self.logger.error(f"❌ Forward pass gagal: {str(e)}")
            raise BackboneError(f"Forward pass gagal: {str(e)}")