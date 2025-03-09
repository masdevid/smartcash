# File: smartcash/handlers/model/core/backbone_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk pembuatan backbone dengan berbagai arsitektur

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any
from pathlib import Path
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError

class BackboneFactory:
    """
    Factory untuk pembuatan backbone dengan berbagai arsitektur.
    Implementasi ulang dari BackboneHandler sebagai factory yang lebih modular.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi backbone factory.
        
        Args:
            config: Konfigurasi model dan backbone
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("backbone_factory")
    
    def create_backbone(
        self, 
        backbone_type: str = 'efficientnet',
        pretrained: bool = True,
        weights_path: Optional[str] = None
    ) -> nn.Module:
        """
        Buat backbone dengan tipe tertentu.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            pretrained: Load pretrained weights
            weights_path: Custom weights path (override pretrained)
            
        Returns:
            Backbone yang diinisialisasi
        """
        self.logger.info(f"🔄 Membuat backbone {backbone_type} (pretrained={pretrained})")
        
        backbone_type = backbone_type.lower()
        
        try:
            if backbone_type == 'efficientnet':
                backbone = self._create_efficientnet(pretrained)
            elif backbone_type == 'cspdarknet':
                backbone = self._create_cspdarknet(pretrained)
            else:
                raise ModelError(f"Tipe backbone {backbone_type} tidak didukung")
        
            # Load weights kustom jika diberikan
            if weights_path:
                self._load_weights(backbone, weights_path)
                
            self.logger.success(f"✅ Backbone {backbone_type} berhasil dibuat")
            return backbone
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backbone {backbone_type}: {str(e)}")
            raise ModelError(f"Gagal membuat backbone: {str(e)}")
    
    def _create_efficientnet(self, pretrained: bool = True) -> nn.Module:
        """
        Buat EfficientNet backbone.
        
        Args:
            pretrained: Load pretrained weights
            
        Returns:
            EfficientNet backbone
        """
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=weights)
        
        # Hapus classifier untuk deteksi
        model.classifier = nn.Identity()
        
        return model
    
    def _create_cspdarknet(self, pretrained: bool = True) -> nn.Module:
        """
        Buat CSPDarknet backbone.
        
        Args:
            pretrained: Load pretrained weights
            
        Returns:
            CSPDarknet backbone
        """
        try:
            # Import CSPDarknet dari YOLOv5
            from smartcash.models.yolov5_components import CSPDarknet
            
            # Buat CSPDarknet
            model = CSPDarknet(
                depth_multiple=1.0,
                width_multiple=1.0,
                pretrained=pretrained
            )
            
            return model
            
        except ImportError:
            self.logger.error("❌ CSPDarknet tidak tersedia. Pastikan YOLOv5 components diimplementasikan")
            raise ModelError("CSPDarknet backbone tidak tersedia")
    
    def _load_weights(self, model: nn.Module, weights_path: str) -> None:
        """
        Load weights ke backbone.
        
        Args:
            model: Model backbone
            weights_path: Path ke file weights
            
        Raises:
            ModelError: Jika gagal memuat weights
        """
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise ModelError(f"File weights tidak ditemukan: {weights_path}")
        
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            self.logger.success(f"✅ Weights berhasil dimuat dari {weights_path}")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat weights: {str(e)}")
            raise ModelError(f"Gagal memuat weights: {str(e)}")
    
    def get_feature_dim(self, backbone_type: str = 'efficientnet') -> int:
        """
        Dapatkan dimensi fitur output dari backbone.
        
        Args:
            backbone_type: Tipe backbone
            
        Returns:
            Dimensi fitur output
        """
        backbone_type = backbone_type.lower()
        
        if backbone_type == 'efficientnet':
            return 1792  # EfficientNet-B4 feature dimension
        elif backbone_type == 'cspdarknet':
            return 1024  # CSPDarknet feature dimension
        else:
            raise ModelError(f"Tipe backbone {backbone_type} tidak didukung")