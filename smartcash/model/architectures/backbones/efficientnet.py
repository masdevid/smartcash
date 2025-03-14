"""
File: smartcash/model/architectures/backbones/efficientnet.py
Deskripsi: Implementasi EfficientNet backbone untuk YOLOv5 dengan adaptasi channel
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
import timm

from smartcash.common.logger import SmartCashLogger
from smartcash.model.exceptions import BackboneError
from smartcash.model.architectures.backbones.base import BaseBackbone

class EfficientNetBackbone(BaseBackbone):
    """
    EfficientNet backbone untuk arsitektur YOLOv5.
    
    Menggunakan pretrained EfficientNet dari library timm dengan
    adaptasi channel output untuk memastikan kompatibilitas dengan
    arsitektur YOLOv5.
    """
    
    # Channel yang diharapkan dari berbagai varian EfficientNet
    EXPECTED_CHANNELS = {
        'efficientnet_b0': [24, 48, 208],  # P3, P4, P5 stages
        'efficientnet_b1': [32, 88, 320],
        'efficientnet_b2': [32, 112, 352], 
        'efficientnet_b3': [40, 112, 384],
        'efficientnet_b4': [56, 160, 448],
        'efficientnet_b5': [64, 176, 512],
        'efficientnet_b6': [72, 200, 576],
        'efficientnet_b7': [80, 224, 640],
    }
    
    # Output channels standar yang digunakan YOLOv5
    YOLO_CHANNELS = [128, 256, 512]
    
    def __init__(
        self, 
        pretrained: bool = True, 
        model_name: str = 'efficientnet_b4',
        out_indices: Tuple[int, ...] = (2, 3, 4),  # P3, P4, P5 stages
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi EfficientNet backbone.
        
        Args:
            pretrained: Gunakan pretrained weights atau tidak
            model_name: Nama model EfficientNet (efficientnet_b0 hingga efficientnet_b7)
            out_indices: Indeks untuk output feature map
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
            
            # Buat adapter layer untuk konversi channel
            self.adapters = nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
                for in_ch, out_ch in zip(actual_channels, self.YOLO_CHANNELS)
            ])
            
            self.logger.success(
                f"✅ Berhasil load EfficientNet backbone:\n"
                f"   • Model: {model_name}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Input channels: {actual_channels}\n"
                f"   • Output channels: {self.YOLO_CHANNELS}"
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
            for i, feat in enumerate(features):
                batch, channels, height, width = feat.shape
                expected_channels = self.actual_channels[i]
                
                if channels != expected_channels:
                    self.logger.warning(
                        f"⚠️ Feature {i} memiliki {channels} channels, "
                        f"namun yang diharapkan {expected_channels} channels!"
                    )
            
            # Apply channel adapters
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
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        """
        Load weights dari state dictionary.
        
        Args:
            state_dict: State dictionary dengan weights
            strict: Flag untuk strict loading
            
        Raises:
            BackboneError: Jika loading weights gagal
        """
        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=strict
            )
            
            if missing_keys and self.logger:
                self.logger.warning(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys and self.logger:
                self.logger.warning(f"⚠️ Unexpected keys: {unexpected_keys}")
                
            self.logger.success("✅ Berhasil memuat weights kustom")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat weights: {str(e)}")
            raise BackboneError(f"Gagal memuat weights: {str(e)}")