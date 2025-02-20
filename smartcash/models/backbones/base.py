# File: models/backbones/base.py
# Author: Alfrida Sabar
# Deskripsi: Abstract base class untuk backbone YOLOv5

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple

from smartcash.utils.logger import SmartCashLogger

class BaseBackbone(ABC, nn.Module):
    """Base class for all backbone networks."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_output_channels(self) -> List[int]:
        """Get the number of output channels for each feature level."""
        pass
    
    @abstractmethod
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get the spatial dimensions of the output feature maps."""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Forward pass of the backbone network."""
        pass