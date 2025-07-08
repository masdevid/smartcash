"""
File: smartcash/model/core/model_builder.py
Deskripsi: Builder untuk konstruksi model SmartCash dengan berbagai backbone options
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.utils.progress_bridge import ModelProgressBridge
from smartcash.model.utils.backbone_factory import BackboneFactory
from smartcash.model.core.yolo_head import YOLOHead

class ModelBuilder:
    """🏗️ Builder untuk konstruksi model SmartCash dengan backbone selection"""
    
    def __init__(self, config: Dict[str, Any], progress_bridge: ModelProgressBridge):
        self.config = config
        self.progress_bridge = progress_bridge
        self.logger = get_logger("model.builder")
        self.backbone_factory = BackboneFactory()
        
    def build(self, backbone: str = 'efficientnet_b4', detection_layers: List[str] = None, 
              layer_mode: str = 'single', num_classes: int = 7, img_size: int = 640,
              feature_optimization: Dict = None, **kwargs) -> nn.Module:
        """🔧 Build complete model dengan konfigurasi yang diberikan"""
        
        try:
            detection_layers = detection_layers or ['banknote']
            feature_optimization = feature_optimization or {'enabled': False}
            
            self.logger.info(f"🏗️ Building {backbone} model | Layers: {detection_layers} | Mode: {layer_mode}")
            
            # Step 1: Build backbone
            self.progress_bridge.update_substep(1, 4, f"🔧 Building {backbone} backbone...")
            backbone_model = self.backbone_factory.create_backbone(
                backbone, 
                pretrained=True,
                feature_optimization=feature_optimization.get('enabled', False)
            )
            
            # Step 2: Build neck (FPN-PAN)
            self.progress_bridge.update_substep(2, 4, "🔗 Building neck (FPN-PAN)...")
            neck = self._build_neck(backbone_model.get_output_channels(), feature_optimization)
            
            # Step 3: Build detection head
            self.progress_bridge.update_substep(3, 4, f"🎯 Building detection head for {len(detection_layers)} layers...")
            head = self._build_detection_head(
                neck.get_output_channels(),
                detection_layers=detection_layers,
                layer_mode=layer_mode,
                num_classes=num_classes,
                img_size=img_size
            )
            
            # Step 4: Assemble complete model
            self.progress_bridge.update_substep(4, 4, "🔗 Assembling complete model...")
            model = SmartCashYOLO(
                backbone=backbone_model,
                neck=neck,
                head=head,
                config={
                    'backbone': backbone,
                    'detection_layers': detection_layers,
                    'layer_mode': layer_mode,
                    'num_classes': num_classes,
                    'img_size': img_size
                }
            )
            
            self.logger.info(f"✅ Model built successfully | Params: {self._count_parameters(model):,}")
            return model
            
        except Exception as e:
            error_msg = f"❌ Model building failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _build_neck(self, backbone_channels: List[int], feature_optimization: Dict) -> nn.Module:
        """Build FPN-PAN neck"""
        from smartcash.model.architectures.necks.fpn_pan import FPN_PAN
        
        # Output channels untuk YOLOv5 compatibility
        out_channels = [128, 256, 512]  # P3, P4, P5
        
        neck = FPN_PAN(
            in_channels=backbone_channels,
            out_channels=out_channels
        )
        
        self.logger.debug(f"🔗 Neck built: {backbone_channels} → {out_channels}")
        return neck
    
    def _build_detection_head(self, neck_channels: List[int], detection_layers: List[str],
                            layer_mode: str, num_classes: int, img_size: int) -> nn.Module:
        """Build YOLO detection head"""
        head = YOLOHead(
            in_channels=neck_channels,
            detection_layers=detection_layers,
            layer_mode=layer_mode,
            num_classes=num_classes,
            img_size=img_size
        )
        
        self.logger.debug(f"🎯 Head built: {layer_mode} mode | {len(detection_layers)} layers")
        return head
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Hitung total parameter model"""
        return sum(p.numel() for p in model.parameters())


class SmartCashYOLO(nn.Module):
    """🎯 Complete SmartCash YOLO model dengan modular architecture"""
    
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.config = config
        self.logger = get_logger("model.yolo")
        
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """Forward pass melalui backbone → neck → head"""
        try:
            # Feature extraction
            features = self.backbone(x)
            
            # Feature processing
            processed_features = self.neck(features)
            
            # Detection
            detections = self.head(processed_features)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ Forward pass error: {str(e)}")
            raise RuntimeError(f"Forward pass failed: {str(e)}")
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.25, 
                nms_threshold: float = 0.45) -> Dict[str, Any]:
        """Prediction dengan post-processing"""
        self.eval()
        with torch.no_grad():
            # Raw predictions
            raw_predictions = self.forward(x)
            
            # Post-processing (NMS, filtering, etc.)
            processed_predictions = self._postprocess_predictions(
                raw_predictions, conf_threshold, nms_threshold
            )
            
            return processed_predictions
    
    def _postprocess_predictions(self, predictions: Dict, conf_threshold: float, 
                               nms_threshold: float) -> Dict[str, Any]:
        """Post-process raw predictions"""
        # Placeholder untuk post-processing logic
        # Will be implemented in training/evaluation phases
        return {
            'raw_predictions': predictions,
            'detections': [],
            'num_detections': 0,
            'confidence_threshold': conf_threshold,
            'nms_threshold': nms_threshold
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Summary informasi model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'SmartCash YOLO',
            'backbone': self.config['backbone'],
            'layer_mode': self.config['layer_mode'],
            'detection_layers': self.config['detection_layers'],
            'num_classes': self.config['num_classes'],
            'img_size': self.config['img_size'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Estimate 4 bytes per param
        }