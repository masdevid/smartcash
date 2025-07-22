"""
File: smartcash/model/utils/backbone_factory.py
Deskripsi: Factory untuk pembuatan backbone models (CSPDarknet vs EfficientNet-B4)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.utils.device_utils import get_device_info

class BackboneFactory:
    """🏭 Factory untuk pembuatan backbone models dengan auto-detection"""
    
    def __init__(self):
        self.logger = get_logger("model.backbone_factory")
        self.available_backbones = {
            'cspdarknet': CSPDarknetBackbone,
            'efficientnet_b4': EfficientNetB4Backbone
        }
    
    def create_backbone(self, backbone_type: str, pretrained: bool = True, 
                       feature_optimization: bool = False, **kwargs) -> 'BackboneBase':
        """🔧 Create backbone berdasarkan type yang diminta"""
        
        if backbone_type not in self.available_backbones:
            raise ValueError(f"❌ Backbone '{backbone_type}' tidak didukung. Available: {list(self.available_backbones.keys())}")
        
        backbone_class = self.available_backbones[backbone_type]
        
        self.logger.info(f"🏗️ Creating {backbone_type} backbone | Pretrained: {pretrained} | Optimization: {feature_optimization}")
        
        backbone = backbone_class(
            pretrained=pretrained,
            feature_optimization=feature_optimization,
            **kwargs
        )
        
        # Log backbone info
        channels = backbone.get_output_channels()
        params = sum(p.numel() for p in backbone.parameters())
        self.logger.info(f"✅ {backbone_type} ready | Channels: {channels} | Params: {params:,}")
        
        return backbone
    
    def list_available_backbones(self) -> List[str]:
        """📋 List backbone yang tersedia"""
        return list(self.available_backbones.keys())


class BackboneBase(nn.Module):
    """🎯 Base class untuk semua backbone implementations"""
    
    def __init__(self):
        super().__init__()
        self.output_channels = []
        self.logger = get_logger(f"model.backbone.{self.__class__.__name__.lower()}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass, return feature maps P3, P4, P5"""
        raise NotImplementedError("Subclass must implement forward method")
    
    def get_output_channels(self) -> List[int]:
        """Return output channels untuk P3, P4, P5"""
        return self.output_channels
    
    def freeze(self) -> None:
        """Freeze backbone parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self.logger.info("❄️ Backbone frozen")
    
    def unfreeze(self) -> None:
        """Unfreeze backbone parameters"""
        for param in self.parameters():
            param.requires_grad = True
        self.logger.info("🔥 Backbone unfrozen")


class CSPDarknetBackbone(BackboneBase):
    """🌑 CSPDarknet backbone untuk YOLOv5 baseline"""
    
    def __init__(self, pretrained: bool = True, feature_optimization: bool = False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.feature_optimization = feature_optimization
        self.output_channels = [128, 256, 512]  # P3, P4, P5 for YOLOv5s
        
        self._build_backbone()
        if pretrained:
            self._load_pretrained_weights()
    
    def _build_backbone(self):
        """Build CSPDarknet architecture"""
        try:
            # Try to use YOLOv5 from ultralytics
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=self.pretrained)
            
            # Extract backbone (first 10 layers)
            self.backbone = nn.Sequential(*list(model.model.children())[:10])
            
            self.logger.info("✅ CSPDarknet from ultralytics/yolov5")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load from hub: {str(e)}, building custom implementation")
            self._build_custom_csp_darknet()
    
    def _build_custom_csp_darknet(self):
        """Build custom CSPDarknet jika hub tidak tersedia"""
        # Simplified CSPDarknet implementation
        self.backbone = nn.Sequential(
            # Initial conv
            self._conv_block(3, 32, 6, 2, 2),
            self._conv_block(32, 64, 3, 2, 1),
            
            # CSP stages
            self._csp_stage(64, 128, 3),   # P3 level
            self._csp_stage(128, 256, 3),  # P4 level
            self._csp_stage(256, 512, 3),  # P5 level
        )
        self.logger.info("✅ Custom CSPDarknet built")
    
    def _conv_block(self, in_ch: int, out_ch: int, k: int, s: int, p: int) -> nn.Sequential:
        """Conv + BatchNorm + SiLU block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def _csp_stage(self, in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
        """CSP stage dengan residual blocks"""
        return nn.Sequential(
            self._conv_block(in_ch, out_ch, 3, 2, 1),
            *[self._residual_block(out_ch) for _ in range(num_blocks)]
        )
    
    def _residual_block(self, channels: int) -> nn.Sequential:
        """Residual block"""
        return nn.Sequential(
            self._conv_block(channels, channels//2, 1, 1, 0),
            self._conv_block(channels//2, channels, 3, 1, 1)
        )
    
    def _load_pretrained_weights(self):
        """Load pretrained weights dari path atau download"""
        pretrained_path = Path('/data/pretrained/yolov5s.pt')
        
        if pretrained_path.exists():
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model'].state_dict()
                else:
                    state_dict = checkpoint
                
                # Load compatible weights
                self.load_state_dict(state_dict, strict=False)
                self.logger.info(f"✅ Pretrained weights loaded from {pretrained_path}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load pretrained weights: {str(e)}")
        else:
            self.logger.info("ℹ️ No pretrained weights found, using random initialization")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass return P3, P4, P5 features"""
        features = []
        
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Extract features at P3, P4, P5 levels
            if i in [4, 6, 8]:  # Adjust indices based on architecture
                features.append(x)
        
        return features


class EfficientNetB4Backbone(BackboneBase):
    """🚀 EfficientNet-B4 backbone untuk enhanced performance"""
    
    def __init__(self, pretrained: bool = True, feature_optimization: bool = False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.feature_optimization = feature_optimization
        self.output_channels = [56, 160, 448]  # EfficientNet-B4 feature channels
        
        self._build_backbone()
        if feature_optimization:
            self._add_feature_optimization()
    
    def _build_backbone(self):
        """Build EfficientNet-B4 backbone"""
        try:
            import timm
            
            # Load EfficientNet-B4 dari timm
            self.backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=self.pretrained,
                features_only=True,
                out_indices=[2, 3, 4]  # P3, P4, P5 levels
            )
            
            self.logger.info("✅ EfficientNet-B4 from timm")
            
        except ImportError:
            self.logger.error("❌ timm library not found. Install with: pip install timm")
            raise RuntimeError("timm library required for EfficientNet-B4")
        except Exception as e:
            self.logger.error(f"❌ Failed to create EfficientNet-B4: {str(e)}")
            raise RuntimeError(f"EfficientNet-B4 creation failed: {str(e)}")
    
    def _add_feature_optimization(self):
        """Add feature optimization layers"""
        self.feature_adapters = nn.ModuleList([
            self._create_adapter(ch, target_ch) 
            for ch, target_ch in zip(self.output_channels, [128, 256, 512])
        ])
        
        # Update output channels untuk compatibility
        self.output_channels = [128, 256, 512]
        self.logger.info("🔧 Feature optimization adapters added")
    
    def _create_adapter(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create feature adapter layer"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            # Channel attention jika optimization enabled
            ChannelAttention(out_ch) if self.feature_optimization else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass return P3, P4, P5 features"""
        # Extract features dari EfficientNet
        features = self.backbone(x)
        
        # Apply feature adapters jika ada
        if hasattr(self, 'feature_adapters'):
            features = [adapter(feat) for feat, adapter in zip(features, self.feature_adapters)]
        
        return features


class ChannelAttention(nn.Module):
    """📡 Channel attention module untuk feature optimization"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return attention


# Factory convenience functions
def create_cspdarknet_backbone(pretrained: bool = True, **kwargs) -> CSPDarknetBackbone:
    """🌑 Create CSPDarknet backbone"""
    return CSPDarknetBackbone(pretrained=pretrained, **kwargs)

def create_efficientnet_backbone(pretrained: bool = True, feature_optimization: bool = False, **kwargs) -> EfficientNetB4Backbone:
    """🚀 Create EfficientNet-B4 backbone"""
    return EfficientNetB4Backbone(pretrained=pretrained, feature_optimization=feature_optimization, **kwargs)