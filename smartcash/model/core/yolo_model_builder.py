"""
File: smartcash/model/core/yolo_model_builder.py
Description: Enhanced YOLO model builder with multi-layer detection support
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import ModelError
from smartcash.model.architectures.backbones.efficientnet import EfficientNetBackbone
from smartcash.model.architectures.backbones.cspdarknet import CSPDarknet
from smartcash.model.architectures.necks.fpn_pan import FPN_PAN
from smartcash.model.architectures.heads.multi_layer_head import MultiLayerHead
from smartcash.model.architectures.heads.detection_head import DetectionHead
from smartcash.model.training.multi_task_loss import UncertaintyMultiTaskLoss, AdaptiveMultiTaskLoss
from smartcash.model.config.model_constants import YOLO_CHANNELS


class SmartCashYOLOv5(nn.Module):
    """
    Enhanced YOLOv5 model with multi-layer detection support for banknote recognition
    
    Architecture:
    - Backbone: EfficientNet-B4 or CSPDarknet
    - Neck: FPN-PAN
    - Heads: Multi-layer detection (layer_1, layer_2, layer_3)
    """
    
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module,
                 model_config: Dict[str, Any], logger: Optional[SmartCashLogger] = None):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.model_config = model_config
        
        self.backbone = backbone
        self.neck = neck
        self.head = head
        
        # Model metadata
        self.architecture_info = self._extract_architecture_info()
        
        self.logger.info(f"✅ SmartCashYOLOv5 model created")
        self.logger.info(f"   • Backbone: {self.architecture_info['backbone']['type']}")
        self.logger.info(f"   • Neck: {self.architecture_info['neck']['type']}")
        self.logger.info(f"   • Head: {self.architecture_info['head']['type']}")
        self.logger.info(f"   • Total parameters: {self.count_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass through the complete model
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dict of layer predictions {layer_name: [pred_p3, pred_p4, pred_p5]}
        """
        # Backbone forward pass
        backbone_features = self.backbone(x)
        
        # Neck forward pass
        neck_features = self.neck(backbone_features)
        
        # Head forward pass
        predictions = self.head(neck_features)
        
        return predictions
    
    def _extract_architecture_info(self) -> Dict[str, Any]:
        """Extract architecture information from components"""
        info = {
            'backbone': {
                'type': self.backbone.__class__.__name__,
                'info': self.backbone.get_info() if hasattr(self.backbone, 'get_info') else {}
            },
            'neck': {
                'type': self.neck.__class__.__name__,
                'info': self.neck.get_info() if hasattr(self.neck, 'get_info') else {}
            },
            'head': {
                'type': self.head.__class__.__name__,
                'info': self.head.get_layer_info() if hasattr(self.head, 'get_layer_info') else {}
            }
        }
        return info
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'architecture': self.architecture_info,
            'model_config': self.model_config,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters(),
            'model_size_mb': self.count_parameters() * 4 / (1024 * 1024),  # Rough estimate
            'input_size': self.model_config.get('input_size', (640, 640))
        }
    
    def prepare_for_training(self, freeze_backbone: bool = True) -> Dict[str, Any]:
        """Prepare model for training with phase-based configuration"""
        if freeze_backbone:
            # Phase 1: Freeze backbone, train heads only
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            phase = 'phase_1'
            self.logger.info("❄️ Phase 1: Backbone frozen, training neck and heads")
        else:
            # Phase 2: Unfreeze entire model for fine-tuning
            for param in self.parameters():
                param.requires_grad = True
            phase = 'phase_2'
            self.logger.info("🔥 Phase 2: Full model unfrozen for fine-tuning")
        
        return {
            'phase': phase,
            'frozen_backbone': freeze_backbone,
            'trainable_parameters': self.count_trainable_parameters(),
            'total_parameters': self.count_parameters()
        }


class YOLOModelBuilder:
    """
    Enhanced YOLO model builder with support for multi-layer detection
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[SmartCashLogger] = None):
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Extract configuration sections
        self.backbone_config = config.get('backbone', {})
        self.neck_config = config.get('neck', {})
        self.head_config = config.get('head', {})
        self.model_config = config.get('model', {})
        
        self.logger.info(f"🏗️ YOLOModelBuilder initialized with config")
    
    def build_model(self, testing_mode: bool = False) -> Dict[str, Any]:
        """
        Build complete YOLO model with specified configuration
        
        Args:
            testing_mode: Whether to build in testing mode (dummy components)
            
        Returns:
            Dict containing model and build information
        """
        try:
            self.logger.info(f"🚀 Building YOLO model (testing_mode={testing_mode})")
            
            # Build backbone
            backbone_result = self._build_backbone(testing_mode)
            if not backbone_result['success']:
                raise ModelError(f"Failed to build backbone: {backbone_result['error']}")
            backbone = backbone_result['component']
            
            # Build neck
            neck_result = self._build_neck(backbone.out_channels, testing_mode)
            if not neck_result['success']:
                raise ModelError(f"Failed to build neck: {neck_result['error']}")
            neck = neck_result['component']
            
            # Build head
            head_result = self._build_head(neck.out_channels, testing_mode)
            if not head_result['success']:
                raise ModelError(f"Failed to build head: {head_result['error']}")
            head = head_result['component']
            
            # Create complete model
            model = SmartCashYOLOv5(
                backbone=backbone,
                neck=neck,
                head=head,
                model_config=self.model_config,
                logger=self.logger
            )
            
            # Build loss function
            loss_result = self._build_loss_function()
            
            build_info = {
                'backbone_info': backbone_result['info'],
                'neck_info': neck_result['info'],
                'head_info': head_result['info'],
                'loss_info': loss_result,
                'model_info': model.get_model_info(),
                'build_config': self.config,
                'testing_mode': testing_mode
            }
            
            self.logger.info(f"✅ YOLO model built successfully")
            self.logger.info(f"   • Total parameters: {model.count_parameters():,}")
            
            return {
                'success': True,
                'model': model,
                'loss_function': loss_result.get('loss_function'),
                'build_info': build_info
            }
            
        except Exception as e:
            error_msg = f"Failed to build YOLO model: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'exception': str(e)
            }
    
    def _build_backbone(self, testing_mode: bool = False) -> Dict[str, Any]:
        """Build backbone component"""
        try:
            backbone_type = self.backbone_config.get('type', 'efficientnet_b4')
            
            if backbone_type.startswith('efficientnet'):
                backbone = EfficientNetBackbone(
                    model_name=backbone_type,
                    pretrained=self.backbone_config.get('pretrained', True),
                    testing_mode=testing_mode,
                    multi_layer_heads=self.head_config.get('multi_layer', False),
                    logger=self.logger
                )
            elif backbone_type == 'cspdarknet':
                backbone = CSPDarknet(
                    pretrained=self.backbone_config.get('pretrained', True),
                    model_size=self.backbone_config.get('model_size', 'yolov5s'),
                    testing_mode=testing_mode,
                    multi_layer_heads=self.head_config.get('multi_layer', False),
                    logger=self.logger
                )
            else:
                raise ModelError(f"Unsupported backbone type: {backbone_type}")
            
            return {
                'success': True,
                'component': backbone,
                'info': backbone.get_info()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'component': None,
                'info': {}
            }
    
    def _build_neck(self, in_channels: List[int], testing_mode: bool = False) -> Dict[str, Any]:
        """Build neck component"""
        try:
            neck_type = self.neck_config.get('type', 'fpn_pan')
            
            if neck_type == 'fpn_pan':
                neck = FPN_PAN(
                    in_channels=in_channels,
                    out_channels=self.neck_config.get('out_channels', YOLO_CHANNELS),
                    logger=self.logger
                )
            else:
                raise ModelError(f"Unsupported neck type: {neck_type}")
            
            return {
                'success': True,
                'component': neck,
                'info': neck.get_info() if hasattr(neck, 'get_info') else {}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'component': None,
                'info': {}
            }
    
    def _build_head(self, in_channels: List[int], testing_mode: bool = False) -> Dict[str, Any]:
        """Build detection head component"""
        try:
            use_multi_layer = self.head_config.get('multi_layer', False)
            
            if use_multi_layer:
                # Multi-layer detection head
                layer_specs = {
                    'layer_1': {
                        'description': 'Full banknote detection (main object)',
                        'classes': ['001', '002', '005', '010', '020', '050', '100'],
                        'num_classes': 7
                    },
                    'layer_2': {
                        'description': 'Nominal-defining features (unique visual cues)',
                        'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                        'num_classes': 7
                    },
                    'layer_3': {
                        'description': 'Common features (shared among notes)',
                        'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                        'num_classes': 3
                    }
                }
                
                head = MultiLayerHead(
                    in_channels=in_channels,
                    layer_specs=layer_specs,
                    num_anchors=self.head_config.get('num_anchors', 3),
                    img_size=self.model_config.get('img_size', 640),
                    use_attention=self.head_config.get('use_attention', True),
                    logger=self.logger
                )
                
                head_info = head.get_layer_info()
            else:
                # Single layer detection head
                head = DetectionHead(
                    in_channels=in_channels,
                    detection_layers=['banknote'],
                    num_classes=7,
                    layer_mode='single',
                    img_size=self.model_config.get('img_size', 640),
                    use_attention=self.head_config.get('use_attention', False),
                    logger=self.logger
                )
                
                head_info = head.get_config()
            
            return {
                'success': True,
                'component': head,
                'info': head_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'component': None,
                'info': {}
            }
    
    def _build_loss_function(self) -> Dict[str, Any]:
        """Build loss function for training"""
        try:
            use_multi_layer = self.head_config.get('multi_layer', False)
            loss_config = self.config.get('loss', {})
            
            if use_multi_layer:
                # Multi-task loss with uncertainty weighting
                layer_config = {
                    'layer_1': {'description': 'Full banknote detection', 'num_classes': 7},
                    'layer_2': {'description': 'Nominal-defining features', 'num_classes': 7},
                    'layer_3': {'description': 'Common features', 'num_classes': 3}
                }
                
                use_adaptive = loss_config.get('adaptive', False)
                if use_adaptive:
                    loss_function = AdaptiveMultiTaskLoss(
                        layer_config=layer_config,
                        loss_config=loss_config,
                        logger=self.logger
                    )
                else:
                    loss_function = UncertaintyMultiTaskLoss(
                        layer_config=layer_config,
                        loss_config=loss_config,
                        logger=self.logger
                    )
                
                loss_type = 'AdaptiveMultiTaskLoss' if use_adaptive else 'UncertaintyMultiTaskLoss'
            else:
                # Standard YOLO loss
                from smartcash.model.training.loss_manager import YOLOLoss
                loss_function = YOLOLoss(
                    num_classes=7,
                    box_weight=loss_config.get('box_weight', 0.05),
                    obj_weight=loss_config.get('obj_weight', 1.0),
                    cls_weight=loss_config.get('cls_weight', 0.5)
                )
                loss_type = 'YOLOLoss'
            
            return {
                'success': True,
                'loss_function': loss_function,
                'loss_type': loss_type,
                'loss_config': loss_config
            }
            
        except Exception as e:
            self.logger.error(f"Failed to build loss function: {e}")
            return {
                'success': False,
                'error': str(e),
                'loss_function': None,
                'loss_type': None
            }


# Factory functions
def create_yolo_model_builder(config: Dict[str, Any], **kwargs) -> YOLOModelBuilder:
    """Factory function to create YOLO model builder"""
    return YOLOModelBuilder(config, **kwargs)


def build_banknote_detection_model(backbone_type: str = 'efficientnet_b4',
                                  multi_layer: bool = True,
                                  testing_mode: bool = False,
                                  **kwargs) -> Dict[str, Any]:
    """
    One-liner function to build banknote detection model
    
    Args:
        backbone_type: Type of backbone ('efficientnet_b4' or 'cspdarknet')
        multi_layer: Whether to use multi-layer detection
        testing_mode: Whether to build in testing mode
        **kwargs: Additional configuration parameters
        
    Returns:
        Build result with model and information
    """
    config = {
        'backbone': {
            'type': backbone_type,
            'pretrained': not testing_mode
        },
        'neck': {
            'type': 'fpn_pan',
            'out_channels': YOLO_CHANNELS
        },
        'head': {
            'multi_layer': multi_layer,
            'use_attention': True,
            'num_anchors': 3
        },
        'model': {
            'img_size': 640
        },
        'loss': {
            'box_weight': 0.05,
            'obj_weight': 1.0,
            'cls_weight': 0.5,
            'dynamic_weighting': multi_layer,
            'adaptive': False
        }
    }
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if key in config:
            config[key].update(value if isinstance(value, dict) else {key: value})
    
    builder = YOLOModelBuilder(config)
    return builder.build_model(testing_mode=testing_mode)


def create_efficientnet_yolo(multi_layer: bool = True, testing_mode: bool = False) -> Dict[str, Any]:
    """Create YOLO model with EfficientNet-B4 backbone"""
    return build_banknote_detection_model('efficientnet_b4', multi_layer, testing_mode)


def create_cspdarknet_yolo(multi_layer: bool = True, testing_mode: bool = False) -> Dict[str, Any]:
    """Create YOLO model with CSPDarknet backbone"""
    return build_banknote_detection_model('cspdarknet', multi_layer, testing_mode)