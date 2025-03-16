"""
File: smartcash/model/architectures/backbones/base.py
Deskripsi: Base class for all backbone networks
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple

from smartcash.common.logger import SmartCashLogger
from smartcash.model.exceptions import BackboneError

class BaseBackbone(ABC, nn.Module):
    """
    Kelas dasar untuk semua backbone network.
    
    Menyediakan interface standar yang harus diimplementasikan oleh semua
    backbone networks untuk menjamin kompatibilitas dengan arsitektur YOLOv5.
    """
    
    def __init__(self, logger=None):
        """
        Inisialisasi backbone network.
        
        Args:
            logger: Logger untuk mencatat proses (opsional)
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
    
    @abstractmethod
    def get_output_channels(self) -> List[int]:
        """
        Dapatkan jumlah output channel untuk setiap level feature.
        
        Returns:
            List[int]: Jumlah channel untuk setiap output feature map
        """
        pass
    
    @abstractmethod
    def get_output_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> List[Tuple[int, int]]:
        """
        Dapatkan dimensi spasial dari output feature maps.
        
        Args:
            input_size: Ukuran input (width, height)
            
        Returns:
            List[Tuple[int, int]]: Ukuran spasial untuk setiap output feature map
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass dari backbone network.
        
        Args:
            x: Input tensor dengan shape [batch_size, channels, height, width]
            
        Returns:
            List[torch.Tensor]: Feature maps dari backbone network
        """
        pass
    
    def validate_output(self, features: List[torch.Tensor], expected_channels: List[int] = None) -> bool:
        """
        Validasi output feature maps dari backbone.
        
        Args:
            features: Feature maps dari backbone network
            expected_channels: Channel yang diharapkan (opsional)
            
        Returns:
            bool: True jika validasi berhasil, False jika tidak
            
        Raises:
            BackboneError: Jika output feature maps tidak valid
        """
        try:
            # Validasi jumlah feature maps
            if not features or len(features) < 3:
                raise BackboneError(
                    f"❌ Jumlah feature maps ({len(features) if features else 0}) "
                    f"tidak mencukupi, minimal diperlukan 3 feature maps"
                )
            
            # Validasi dimensi feature maps
            for i, feat in enumerate(features):
                if not isinstance(feat, torch.Tensor):
                    raise BackboneError(f"❌ Feature map {i} bukan torch.Tensor")
                
                if len(feat.shape) != 4:
                    raise BackboneError(
                        f"❌ Feature map {i} memiliki dimensi {len(feat.shape)}, "
                        f"seharusnya 4 (batch, channels, height, width)"
                    )
            
            # Validasi channel jika expected_channels diberikan
            if expected_channels:
                actual_channels = [feat.shape[1] for feat in features]
                if len(actual_channels) != len(expected_channels):
                    self.logger.warning(
                        f"⚠️ Jumlah level feature ({len(actual_channels)}) "
                        f"tidak sesuai dengan yang diharapkan ({len(expected_channels)})"
                    )
                
                for i, (actual, expected) in enumerate(zip(actual_channels, expected_channels)):
                    if actual != expected:
                        self.logger.warning(
                            f"⚠️ Feature level {i} memiliki {actual} channels, "
                            f"sementara yang diharapkan {expected} channels"
                        )
            
            return True
            
        except BackboneError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"❌ Validasi output backbone gagal: {str(e)}")
            raise BackboneError(f"Validasi output backbone gagal: {str(e)}")