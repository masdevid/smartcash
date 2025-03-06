# File: smartcash/models/backbones/efficientnet.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi EfficientNet backbone untuk YOLOv5 dengan validasi dimensi output

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import timm

from .base import BaseBackbone
from smartcash.utils.logger import SmartCashLogger

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet-B4 backbone implementation dengan adaptasi untuk YOLOv5."""
    
    # ✨ PERUBAHAN: Definisi channel yang diharapkan
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
    
    # ✨ PERUBAHAN: Output channels yang diharapkan YOLOv5
    YOLO_CHANNELS = [128, 256, 512]
    
    def __init__(
        self, 
        pretrained: bool = True, 
        model_name: str = 'efficientnet_b4',
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.model_name = model_name
        
        try:
            # Validasi model name
            if model_name not in self.EXPECTED_CHANNELS:
                supported = list(self.EXPECTED_CHANNELS.keys())
                raise ValueError(f"❌ Model {model_name} tidak didukung. Model yang didukung: {supported}")
            
            # Load pretrained EfficientNet
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(2, 3, 4)  # P3, P4, P5 stages
            )
            
            # Dapatkan spesifikasi output channels dari model
            # ✨ PERUBAHAN: Validasi output channel secara dinamis
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 640, 640)
                outputs = self.model(dummy_input)
                actual_channels = [o.shape[1] for o in outputs]
                
                # Log spesifikasi channel yang sebenarnya vs yang diharapkan
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
            
            # Adapter layers to convert EfficientNet output channels to YOLO expected channels
            self.adapters = nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
                for in_ch, out_ch in zip(actual_channels, self.YOLO_CHANNELS)
            ])
            
            self.logger.info(
                f"✅ Berhasil load EfficientNet backbone:\n"
                f"   • Model: {model_name}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Input channels: {actual_channels}\n"
                f"   • Output channels: {self.YOLO_CHANNELS}"
            )
        except Exception as e:
            self.logger.error(f"❌ Gagal load {model_name}: {str(e)}")
            raise
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for P3, P4, P5."""
        # Return adapted channel sizes to match YOLOv5 expectations
        return self.YOLO_CHANNELS
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get output shapes for feature maps dengan input 640x640."""
        return [(80, 80), (40, 40), (20, 20)]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass dengan channel adaptation dan validasi shape."""
        # Get multi-scale features dari EfficientNet
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
            
        return adapted_features
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights dari state dictionary"""
        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
            
            if missing_keys:
                self.logger.warning(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"⚠️ Unexpected keys: {unexpected_keys}")
                
            self.logger.info("✅ Berhasil memuat weights kustom")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat weights: {str(e)}")
            raise