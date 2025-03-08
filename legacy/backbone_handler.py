"""Handler untuk model backbone detection."""

import os
from typing import Dict, Optional, Tuple
import torch
import yaml
from pathlib import Path
from torch import nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from smartcash.utils.logger import SmartCashLogger

class BackboneHandler:
    """Handler untuk model backbone detection."""
    
    def __init__(
        self,
        config_path: str,
        model_type: str = 'efficientnet',
        weights: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize backbone handler.
        
        Args:
            config_path: Path ke file konfigurasi
            model_type: Tipe model backbone ('efficientnet', 'resnet', etc)
            weights: Optional path ke file weights
            logger: Optional logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.model_type = model_type.lower()
        self.weights_path = weights
        
        # Initialize model
        self.model = self._initialize_model()
        if self.weights_path:
            self._load_weights()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['model']
    
    def _initialize_model(self) -> nn.Module:
        """Initialize backbone model."""
        if self.model_type == 'efficientnet':
            model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            # Remove classifier for detection
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        
        return model
    
    def _load_weights(self) -> None:
        """Load pre-trained weights."""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Weight file tidak ditemukan: {self.weights_path}"
            )
        
        try:
            state_dict = torch.load(self.weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.logger.success(f"Weights berhasil dimuat dari {self.weights_path}")
        except Exception as e:
            self.logger.error(f"Gagal memuat weights: {str(e)}")
            raise
    
    def get_model(self) -> nn.Module:
        """Get backbone model."""
        return self.model
    
    def get_feature_dim(self) -> int:
        """Get feature dimension of backbone."""
        if self.model_type == 'efficientnet':
            return 1792  # EfficientNet-B4 feature dimension
        return -1
    
    def save_weights(self, save_path: str) -> None:
        """Save model weights."""
        try:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save(self.model.state_dict(), save_path)
            self.logger.success(f"Weights berhasil disimpan ke {save_path}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan weights: {str(e)}")
            raise
    
    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        self.model = self.model.to(device)
    
    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()
    
    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'type': self.model_type,
            'feature_dim': self.get_feature_dim(),
            'weights': self.weights_path
        }