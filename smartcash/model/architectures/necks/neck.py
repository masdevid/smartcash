import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List
from ultralytics.nn.modules import C2f, SPPF, Concat
from smartcash.common.logger import get_logger

class BaseNeck(nn.Module, ABC):
    """Abstract base class for YOLOv5 neck (PANet)"""
    
    def __init__(self, feature_dims: List[int]):
        super().__init__()
        self.logger = get_logger(__name__)
        self.feature_dims = feature_dims
        self.neck = self.create_neck()
    
    @abstractmethod
    def create_neck(self) -> nn.ModuleList:
        """Create the neck layers"""
        pass
    
    @abstractmethod
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through neck"""
        pass