import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List
from smartcash.common.logger import get_logger

class BaseHead(nn.Module, ABC):
    """Abstract base class for YOLOv5 detection head"""
    
    def __init__(self, num_classes: int, feature_dims: List[int]):
        super().__init__()
        self.logger = get_logger(__name__)
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.head = self.create_head()
    
    @abstractmethod
    def create_head(self) -> nn.Module:
        """Create the detection head"""
        pass
    
    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through head"""
        pass