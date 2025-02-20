# File: models/backbones/cspdarknet.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi CSPDarknet backbone menggunakan pretrained YOLOv5

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from .base import BaseBackbone
from smartcash.utils.logger import SmartCashLogger

class CSPDarknet(BaseBackbone):
    """CSPDarknet backbone implementation."""
    
    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        try:
            # Setup pretrained model dir
            pretrained_dir = Path('./pretrained')
            pretrained_dir.mkdir(exist_ok=True)
            torch.hub.set_dir(str(pretrained_dir))
            
            # Load YOLOv5 model
            yolo = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path='yolov5s.pt',
                trust_repo=True
            )
            
            # Extract backbone layers
            modules = list(yolo.model.model.children())
            backbone_modules = []
            
            for m in modules:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                            backbone_modules.append(layer)
                elif isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    backbone_modules.append(m)
                    
                # Stop after collecting backbone layers
                if len(backbone_modules) >= 10:  # Backbone is first 10 layers
                    break
            
            # Create backbone Sequential
            self.backbone = nn.Sequential(*backbone_modules)
            
            # Extract intermediate feature indices
            self.feature_indices = [4, 6, 9]  # P3, P4, P5 layers
            
            self.logger.success(
                "✨ CSPDarknet backbone berhasil diinisialisasi:\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Weights dir: {pretrained_dir}\n"
                f"   • Feature layers: {self.feature_indices}\n"
                f"   • Channels: {self.get_output_channels()}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Gagal inisialisasi CSPDarknet: {str(e)}")
            raise
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass, return P3, P4, P5 feature maps."""
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for P3, P4, P5."""
        return [128, 256, 512]  # YOLOv5s channels
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get output shapes for feature maps with 640x640 input."""
        return [(80, 80), (40, 40), (20, 20)]
        
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load state dictionary."""
        try:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                self.logger.warning(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"⚠️ Unexpected keys: {unexpected_keys}")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat weights: {str(e)}")
            raise