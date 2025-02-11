# File: src/models/factory.py
# Author: Alfrida Sabar
# Deskripsi: Pabrik model untuk SmartCash Detector dengan dukungan berbagai arsitektur

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn

from .detector import SmartCashDetector
from .efficientnet_backbone import EfficientNetBackbone, SmartCashYOLODetector

class ModelType(Enum):
    """Enum untuk tipe model yang didukung"""
    YOLO_DARKNET = "YOLOv5 with CSPDarknet"
    YOLO_EFFICIENT = "YOLOv5 with EfficientNet-B4"
    YOLO_EFFICIENT_SMALL = "YOLOv5 with EfficientNet-B0"
    YOLO_EFFICIENT_LARGE = "YOLOv5 with EfficientNet-B7"
@dataclass
class ModelConfig:
    """Konfigurasi untuk pembuatan model"""
    type: Optional[ModelType] = None
    weights_path: Optional[str] = None
    img_size: int = 640
    nc: int = 7  # Number of classes (currency denominations)

class ModelFactory:
    """Pabrik untuk membuat model dengan konfigurasi yang berbeda"""
    def create_model(self, config: ModelConfig) -> SmartCashDetector:
        """
        Buat model berdasarkan konfigurasi yang diberikan
        
        Args:
            config (ModelConfig): Konfigurasi model
        
        Returns:
            SmartCashDetector: Model yang dibuat
        """
        # Pilih backbone berdasarkan tipe model
        backbone = None
        if config.type == ModelType.YOLO_EFFICIENT:
            backbone = EfficientNetBackbone(phi=4)  # EfficientNet-B4
        elif config.type == ModelType.YOLO_EFFICIENT_SMALL:
            backbone = EfficientNetBackbone(phi=0)  # EfficientNet-B0
        elif config.type == ModelType.YOLO_EFFICIENT_LARGE:
            backbone = EfficientNetBackbone(phi=7)  # EfficientNet-B7
        
        # Buat detector dengan backbone yang dipilih
        return SmartCashYOLODetector(
            backbone=backbone,
            weights_path=config.weights_path,
            img_size=config.img_size,
            nc=config.nc
        )