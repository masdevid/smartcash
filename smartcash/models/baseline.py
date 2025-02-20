"""Baseline model untuk SmartCash."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class BaselineModel(nn.Module):
    """Model baseline untuk deteksi uang kertas."""
    
    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = 'efficientnet',
        pretrained: bool = True,
        img_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize baseline model.
        
        Args:
            num_classes: Jumlah kelas untuk deteksi
            backbone: Tipe backbone ('efficientnet', 'resnet', etc)
            pretrained: Gunakan pretrained weights atau tidak
            img_size: Ukuran input image (width, height)
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_type = backbone
        self.img_size = img_size
        
        # Initialize backbone
        if backbone == 'efficientnet':
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b4(weights=weights)
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1792
        else:
            raise ValueError(f"Backbone {backbone} tidak didukung")
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * 5, kernel_size=1)  # 5 = [x, y, w, h, conf]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Predictions tensor of shape (batch_size, num_classes * 5, H/32, W/32)
        """
        # Extract features
        features = self.backbone.features(x)
        
        # Detection head
        predictions = self.detection_head(features)
        
        # Reshape predictions
        B, _, H, W = predictions.shape
        predictions = predictions.view(B, self.num_classes, 5, H, W)
        predictions = predictions.permute(0, 1, 3, 4, 2)  # (B, num_classes, H, W, 5)
        
        return predictions
    
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions with confidence threshold.
        
        Args:
            x: Input tensor
            conf_threshold: Confidence threshold
            
        Returns:
            List of dictionaries containing predictions for each image
        """
        predictions = self.forward(x)
        batch_detections = []
        
        for pred in predictions:
            # Get confident predictions
            confident_mask = pred[..., 4] > conf_threshold
            detections = pred[confident_mask]
            
            # Create detection dictionary
            detection_dict = {
                'boxes': detections[..., :4],  # [x, y, w, h]
                'scores': detections[..., 4],  # confidence
                'labels': torch.nonzero(confident_mask).squeeze(-1)  # class indices
            }
            batch_detections.append(detection_dict)
        
        return batch_detections
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'num_classes': self.num_classes,
            'backbone': self.backbone_type,
            'img_size': self.img_size,
            'feature_dim': self.feature_dim
        }
