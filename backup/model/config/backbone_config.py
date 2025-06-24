"""
File: smartcash/model/config/backbone_config.py
Deskripsi: Konfigurasi model untuk backbone network (menggunakan konstanta dari model_constants)
"""

from typing import Dict, Any, Optional, List, Union
from smartcash.model.config.model_config import ModelConfig
from smartcash.model.config.model_constants import SUPPORTED_BACKBONES

class BackboneConfig:
    """Konfigurasi untuk backbone networks dengan dukungan khusus untuk EfficientNet."""
    
    # Menggunakan SUPPORTED_BACKBONES dari model_constants.py
    BACKBONE_CONFIGS = SUPPORTED_BACKBONES
    
    def __init__(self, model_config: Optional[ModelConfig] = None, backbone_type: Optional[str] = None):
        """Inisialisasi konfigurasi backbone dengan model_config atau backbone_type."""
        self.model_config = model_config
        
        # Tentukan tipe backbone
        if backbone_type: self.backbone_type = backbone_type
        elif model_config: self.backbone_type = model_config.get('model.backbone', 'efficientnet_b4')
        else: self.backbone_type = 'efficientnet_b4'
        
        # Validasi backbone
        if self.backbone_type not in self.BACKBONE_CONFIGS:
            raise ValueError(f"❌ Backbone '{self.backbone_type}' tidak didukung. Pilih dari: {', '.join(self.BACKBONE_CONFIGS.keys())}")
        
        # Muat konfigurasi backbone
        self.config = self.BACKBONE_CONFIGS[self.backbone_type].copy()
        
        # Update konfigurasi dari model_config jika tersedia
        if model_config and 'backbone' in model_config.config:
            backbone_overrides = model_config.get('backbone', {})
            for key, value in backbone_overrides.items(): self.config[key] = value
    
    @property
    def stride(self) -> int: return self.config['stride']
    
    @property
    def features(self) -> int: return self.config['features']
    
    @property
    def stages(self) -> List[int]: return self.config['stages']
    
    @property
    def pretrained(self) -> bool: return self.config.get('pretrained', True)
    
    @property
    def width_coefficient(self) -> float: return self.config['width_coefficient']
    
    @property
    def depth_coefficient(self) -> float: return self.config['depth_coefficient']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dapatkan nilai konfigurasi backbone berdasarkan key."""
        return self.config.get(key, default)
    
    def is_efficientnet(self) -> bool: return self.backbone_type.startswith('efficientnet')
    
    def get_feature_channels(self) -> List[int]:
        """Dapatkan jumlah channel untuk setiap stage backbone."""
        return self.stages
    
    def to_dict(self) -> Dict[str, Any]:
        """Konversi konfigurasi ke dictionary dengan tipe backbone."""
        return {'type': self.backbone_type, **self.config}