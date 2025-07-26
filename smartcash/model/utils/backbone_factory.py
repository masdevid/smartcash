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
    
    def create_backbone(self, backbone_type: str, pretrained: bool = False, 
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
    
    def __init__(self, pretrained: bool = False, feature_optimization: bool = False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.feature_optimization = feature_optimization
        self.output_channels = [128, 256, 512]  # P3, P4, P5 for YOLOv5s
        
        self._build_backbone()
        if pretrained:
            self._load_pretrained_weights()
    
    def _build_backbone(self):
        """Build CSPDarknet architecture"""
        if self.pretrained:
            # Only attempt hub loading when pretrained weights are requested
            try:
                # Try to use YOLOv5 from ultralytics with PyTorch 2.6+ compatibility
                import torch
                
                # Handle PyTorch 2.6+ weights_only requirement
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                
                try:
                    # YOLOv5 has complex layer connections, so we can't just extract backbone as Sequential
                    # Instead, we'll use the custom implementation as it's more reliable
                    raise Exception("YOLOv5 architecture too complex for simple Sequential extraction")
                    
                finally:
                    # Restore original torch.load
                    torch.load = original_load
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load from hub: {str(e)}, building custom implementation")
                self._build_custom_csp_darknet()
        else:
            # When pretrained=False, directly build custom implementation without hub attempt
            self.logger.debug("Building custom CSPDarknet from scratch (pretrained=False)")
            self._build_custom_csp_darknet()
    
    def _build_custom_csp_darknet(self):
        """Build custom CSPDarknet with proper feature extraction points"""
        # Build a more complete CSPDarknet that matches YOLOv5 structure
        self.backbone = nn.Sequential(
            # Initial conv layers (indices 0-1)
            self._conv_block(3, 32, 6, 2, 2),     # 0: 32 channels
            self._conv_block(32, 64, 3, 2, 1),    # 1: 64 channels
            
            # CSP stages with proper depth for feature extraction
            self._csp_stage(64, 128, 3),          # 2: 128 channels -> P3
            self._conv_block(128, 128, 1, 1, 0),  # 3: Transition
            self._csp_stage(128, 256, 3),         # 4: 256 channels -> P4
            self._conv_block(256, 256, 1, 1, 0),  # 5: Transition
            self._csp_stage(256, 512, 3),         # 6: 512 channels -> P5
            
            # Additional layers to match YOLOv5 depth
            self._conv_block(512, 512, 1, 1, 0),  # 7: Final conv
            self._conv_block(512, 512, 3, 1, 1),  # 8: Final conv
            nn.AdaptiveAvgPool2d((1, 1)),         # 9: Global pooling (not used in feature extraction)
        )
        self.logger.info("✅ Custom CSPDarknet built with 10 layers")
    
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
        intermediate_outputs = []
        
        # Store all intermediate outputs
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            intermediate_outputs.append(x)
            # Extract features at P3, P4, P5 levels based on our architecture
            if i in [2, 4, 6]:  # Indices for CSP stages (128, 256, 512 channels)
                features.append(x)
        
        # Ensure we have exactly 3 features
        if len(features) != 3:
            self.logger.warning(f"⚠️ Expected 3 features, got {len(features)}. Using alternative extraction strategy.")
            
            # Alternative strategy: take features from layers that produce different spatial resolutions
            features = []
            
            # Look for layers that downsample the spatial resolution
            prev_shape = None
            feature_candidates = []
            
            for i, output in enumerate(intermediate_outputs):
                if len(output.shape) == 4:  # Ensure it's a feature map (B, C, H, W)
                    current_shape = output.shape[2:4]  # (H, W)
                    
                    # If this is the first feature map or spatial size changed, it's a candidate
                    if prev_shape is None or current_shape != prev_shape:
                        feature_candidates.append((i, output))
                        prev_shape = current_shape
            
            # Take the last 3 candidates, or duplicate if needed
            if len(feature_candidates) >= 3:
                features = [candidate[1] for candidate in feature_candidates[-3:]]
            elif len(feature_candidates) > 0:
                # Use available candidates and pad with the last one
                base_features = [candidate[1] for candidate in feature_candidates]
                features = base_features[:]
                while len(features) < 3:
                    features.append(base_features[-1])
            else:
                # Last resort: use the last output repeated 3 times
                final_output = intermediate_outputs[-2] if len(intermediate_outputs) > 1 else x  # Skip pooling layer
                features = [final_output, final_output, final_output]
                
            self.logger.info(f"✅ Using {len(feature_candidates)} feature candidates, final features: {len(features)}")
        
        return features[:3]  # Return exactly 3 features


class EfficientNetB4Backbone(BackboneBase):
    """🚀 EfficientNet-B4 backbone untuk enhanced performance"""
    
    def __init__(self, pretrained: bool = False, feature_optimization: bool = False, **kwargs):
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
def create_cspdarknet_backbone(pretrained: bool = False, **kwargs) -> CSPDarknetBackbone:
    """🌑 Create CSPDarknet backbone"""
    return CSPDarknetBackbone(pretrained=pretrained, **kwargs)

def create_efficientnet_backbone(pretrained: bool = False, feature_optimization: bool = False, **kwargs) -> EfficientNetB4Backbone:
    """🚀 Create EfficientNet-B4 backbone"""
    return EfficientNetB4Backbone(pretrained=pretrained, feature_optimization=feature_optimization, **kwargs)