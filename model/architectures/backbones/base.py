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
    """
    Kelas abstrak dasar untuk implementasi backbone model
    
    Semua backbone network harus mewarisi kelas ini dan 
    mengimplementasikan metode yang diperlukan
    """
    
    def __init__(self):
        """Inisialisasi kelas dasar"""
        super().__init__()
        self.out_channels = []  # Jumlah channel output dari setiap level feature
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass backbone untuk mendapatkan feature maps.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List feature maps dari berbagai level backbone
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """
        Dapatkan informasi backbone.
        
        Returns:
            Dictionary berisi informasi backbone
        """
        pass
    
    def load_state_dict_from_path(self, state_dict_path: Union[str, Path], strict: bool = False):
        """
        Muat state dict dari checkpoint custom.
        
        Args:
            state_dict_path: Path ke file state dict
            strict: Apakah menggunakan strict loading
        """
        checkpoint = torch.load(state_dict_path, map_location='cpu')
        
        # Handle berbagai format checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        self.load_state_dict(state_dict, strict=strict)
    
    def freeze(self):
        """Bekukan semua parameter backbone"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Lepaskan pembekuan semua parameter backbone"""
        for param in self.parameters():
            param.requires_grad = True