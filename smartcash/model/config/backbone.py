"""
File: smartcash/model/config/backbone.py
Deskripsi: Konfigurasi untuk backbone networks pada model SmartCash
"""

from typing import Dict, Any, Optional, List, Union
from smartcash.model.config.base import ModelConfig

class BackboneConfig:
    """
    Konfigurasi untuk backbone networks dengan dukungan khusus untuk EfficientNet.
    """
    
    # Konfigurasi default untuk berbagai backbone
    BACKBONE_CONFIGS = {
        'efficientnet_b0': {
            'stride': 32,
            'width_coefficient': 1.0,
            'depth_coefficient': 1.0,
            'pretrained': True,
            'features': 1280,
            'stages': [40, 80, 192, 1280]
        },
        'efficientnet_b1': {
            'stride': 32,
            'width_coefficient': 1.0,
            'depth_coefficient': 1.1,
            'pretrained': True,
            'features': 1280,
            'stages': [40, 80, 192, 1280]
        },
        'efficientnet_b2': {
            'stride': 32,
            'width_coefficient': 1.1,
            'depth_coefficient': 1.2,
            'pretrained': True,
            'features': 1408,
            'stages': [48, 88, 208, 1408]
        },
        'efficientnet_b3': {
            'stride': 32,
            'width_coefficient': 1.2,
            'depth_coefficient': 1.4,
            'pretrained': True,
            'features': 1536,
            'stages': [48, 96, 232, 1536]
        },
        'efficientnet_b4': {
            'stride': 32,
            'width_coefficient': 1.4,
            'depth_coefficient': 1.8,
            'pretrained': True,
            'features': 1792,
            'stages': [56, 112, 272, 1792]
        },
        'efficientnet_b5': {
            'stride': 32,
            'width_coefficient': 1.6,
            'depth_coefficient': 2.2,
            'pretrained': True,
            'features': 2048,
            'stages': [64, 128, 304, 2048]
        },
        'cspdarknet33': {
            'stride': 32,
            'width_coefficient': 1.0,
            'depth_coefficient': 1.0,
            'pretrained': True,
            'features': 1024,
            'stages': [64, 128, 256, 1024]
        },
        'mobilenetv3_large': {
            'stride': 32,
            'width_coefficient': 1.0,
            'depth_coefficient': 1.0,
            'pretrained': True,
            'features': 960,
            'stages': [40, 112, 160, 960]
        }
    }
    
    def __init__(self, model_config: Optional[ModelConfig] = None, backbone_type: Optional[str] = None):
        """
        Inisialisasi konfigurasi backbone.
        
        Args:
            model_config: Instance ModelConfig (opsional)
            backbone_type: Jenis backbone (opsional, misalnya 'efficientnet_b4')
        """
        self.model_config = model_config
        
        # Tentukan tipe backbone
        if backbone_type:
            self.backbone_type = backbone_type
        elif model_config:
            self.backbone_type = model_config.get('model.backbone', 'efficientnet_b4')
        else:
            self.backbone_type = 'efficientnet_b4'
        
        # Validasi backbone
        if self.backbone_type not in self.BACKBONE_CONFIGS:
            raise ValueError(f"❌ Backbone '{self.backbone_type}' tidak didukung. "
                           f"Pilih dari: {', '.join(self.BACKBONE_CONFIGS.keys())}")
        
        # Muat konfigurasi backbone
        self.config = self.BACKBONE_CONFIGS[self.backbone_type].copy()
        
        # Update konfigurasi dari model_config jika tersedia
        if model_config and 'backbone' in model_config.config:
            backbone_overrides = model_config.get('backbone', {})
            for key, value in backbone_overrides.items():
                self.config[key] = value
    
    @property
    def stride(self) -> int:
        """Stride backbone."""
        return self.config['stride']
    
    @property
    def features(self) -> int:
        """Jumlah feature channels."""
        return self.config['features']
    
    @property
    def stages(self) -> List[int]:
        """Feature channels per stage."""
        return self.config['stages']
    
    @property
    def pretrained(self) -> bool:
        """Flag untuk menggunakan bobot pretrained."""
        return self.config['pretrained']
    
    @property
    def width_coefficient(self) -> float:
        """Width coefficient (untuk EfficientNet)."""
        return self.config['width_coefficient']
    
    @property
    def depth_coefficient(self) -> float:
        """Depth coefficient (untuk EfficientNet)."""
        return self.config['depth_coefficient']
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dapatkan nilai konfigurasi backbone.
        
        Args:
            key: Kunci konfigurasi
            default: Nilai default jika kunci tidak ditemukan
            
        Returns:
            Nilai konfigurasi
        """
        return self.config.get(key, default)
    
    def is_efficientnet(self) -> bool:
        """Cek apakah backbone adalah EfficientNet."""
        return self.backbone_type.startswith('efficientnet')
    
    def get_feature_channels(self) -> List[int]:
        """
        Dapatkan jumlah channel untuk setiap stage backbone.
        
        Returns:
            List jumlah channel per stage
        """
        return self.stages
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konversi konfigurasi ke dictionary.
        
        Returns:
            Dictionary konfigurasi
        """
        return {
            'type': self.backbone_type,
            **self.config
        }