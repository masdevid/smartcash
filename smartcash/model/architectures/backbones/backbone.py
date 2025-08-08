import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple
from smartcash.common.logger import get_logger

class BaseBackbone(nn.Module, ABC):
    """Abstract base class for YOLOv5 backbones"""
    
    def __init__(self, num_classes: int, pretrained: bool, device: str):
        super().__init__()
        self.logger = get_logger(__name__)
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        
    @abstractmethod
    def create_backbone(self) -> nn.Module:
        """Create the backbone architecture"""
        pass
    
    @abstractmethod
    def create_neck(self, feature_dims: List[int]) -> nn.ModuleList:
        """Create the neck (PANet) layers"""
        pass
    
    @abstractmethod
    def create_head(self, feature_dims: List[int]) -> nn.Module:
        """Create the detection head"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through backbone, neck, and head"""
        pass
    
    def _setup_phase_1(self, model: nn.Module):
        """Setup Phase 1: Freeze backbone, train head only"""
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        self.logger.info("🔒 Phase 1: Backbone frozen, head trainable")
    
    def setup_phase_2(self, model: nn.Module):
        """Setup Phase 2: Unfreeze backbone, fine-tune entire model"""
        for param in model.parameters():
            param.requires_grad = True
        self.logger.info("🔓 Phase 2: Full model trainable")