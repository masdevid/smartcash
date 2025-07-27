"""
File: smartcash/model/architectures/backbones/base.py
Deskripsi: Kelas dasar untuk implementasi backbone model di SmartCash
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union
from pathlib import Path
from abc import ABC, abstractmethod

class BaseBackbone(nn.Module, ABC):
    """Kelas abstrak dasar untuk implementasi backbone model."""
    
    def __init__(self, logger=None):
        """Inisialisasi kelas dasar dengan opsional logger."""
        super().__init__()
        self.out_channels = []  # Jumlah channel output dari setiap level feature
        self.logger = logger
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass untuk mendapatkan feature maps dari input tensor."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """Dapatkan informasi backbone dalam bentuk dictionary."""
        pass
    
    def load_state_dict_from_path(self, state_dict_path: Union[str, Path], strict: bool = False):
        """Muat state dict dari checkpoint dengan dukungan berbagai format."""
        # Use safe globals for PyTorch 2.6+ compatibility
        import torch.serialization
        try:
            from models.yolo import Model as YOLOModel
            from models.common import Conv, C3, SPPF, Bottleneck
            safe_globals = [YOLOModel, Conv, C3, SPPF, Bottleneck]
        except ImportError:
            safe_globals = []
        
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(state_dict_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint if isinstance(checkpoint, dict) else checkpoint
        self.load_state_dict(state_dict, strict=strict)
    
    def freeze(self):
        """Bekukan semua parameter backbone."""
        for param in self.parameters(): param.requires_grad = False
    
    def unfreeze(self):
        """Lepaskan pembekuan semua parameter backbone."""
        for param in self.parameters(): param.requires_grad = True
        
    def validate_output(self, features: List[torch.Tensor], expected_channels: List[int]):
        """Validasi output feature maps sesuai dengan channel yang diharapkan."""
        if len(features) != len(expected_channels): 
            if self.logger: self.logger.warning(f"⚠️ Jumlah feature maps ({len(features)}) tidak sesuai dengan jumlah expected channels ({len(expected_channels)})")
        for i, (feat, ch) in enumerate(zip(features, expected_channels)):
            if feat.shape[1] != ch and self.logger: 
                self.logger.warning(f"⚠️ Feature map {i} memiliki {feat.shape[1]} channels, diharapkan {ch} channels")