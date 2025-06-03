"""
File: smartcash/model/manager.py
Deskripsi: Simplified model manager dengan training integration yang diperbarui
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.utils.pretrained_model_utils import check_pretrained_model_in_drive, load_pretrained_model
from smartcash.common.exceptions import ModelError, ModelConfigurationError
from smartcash.common.layer_config import get_layer_config
from smartcash.model.config.model_constants import SUPPORTED_BACKBONES, OPTIMIZED_MODELS, get_model_config, DETECTION_LAYERS
from smartcash.model.config import DEFAULT_MODEL_CONFIG

# Imports untuk komponen model
from smartcash.model.architectures.backbones import EfficientNetBackbone, CSPDarknet
from smartcash.model.architectures.necks import FeatureProcessingNeck
from smartcash.model.architectures.heads import DetectionHead
from smartcash.model.components import YOLOLoss
from smartcash.model.models.yolov5_model import YOLOv5Model
from smartcash.model.utils.layer_validator import validate_layer_params, get_num_classes_for_layers

class ModelManager:
    """Simplified Model Manager dengan training integration yang efisien"""
    
    def __init__(self, config: Optional[Dict] = None, model_type: str = 'efficient_optimized', 
                 layer_mode: str = 'single', detection_layers: Optional[List[str]] = None, 
                 pretrained_models_path: str = '/content/drive/MyDrive/SmartCash/models',
                 testing_mode: bool = False, logger = None):
        """Inisialisasi dengan parameter yang telah divalidasi dari UI training"""
        self.logger = logger or get_logger(__name__)
        self.testing_mode = testing_mode
        self.pretrained_models_path = pretrained_models_path
        
        # Load default config dan apply model type
        self.config = DEFAULT_MODEL_CONFIG.copy()
        model_config = OPTIMIZED_MODELS.get(model_type, OPTIMIZED_MODELS['efficient_optimized'])
        self.config.update({**model_config, **(config or {})})
        
        # Validasi dan set layer parameters menggunakan utility yang sama dengan training UI
        self.model_type = model_type
        self.layer_mode, self.detection_layers = validate_layer_params(
            layer_mode or self.config.get('layer_mode', 'single'),
            detection_layers or self.config.get('detection_layers', ['banknote'])
        )
        
        # Update config dengan validated parameters
        self.config.update({'layer_mode': self.layer_mode, 'detection_layers': self.detection_layers, 
                           'num_classes': get_num_classes_for_layers(self.detection_layers) if self.layer_mode == 'multilayer' else self.config.get('num_classes', 7)})
        
        # Inisialisasi komponen model
        self.backbone = self.neck = self.head = self.model = self.loss_fn = None
        self.checkpoint_service = None
        
        self.logger.info(f"✅ ModelManager diinisialisasi: {model_type} | {self.layer_mode} | {self.detection_layers} | {self.config['num_classes']} kelas")
    
    def build_model(self) -> nn.Module:
        """Build model dengan komponennya menggunakan one-liner style"""
        try:
            # Build komponen dengan error handling
            self._build_backbone(), self._build_neck(), self._build_head(), self._build_loss_function()
            
            # Create YOLOv5 model dengan komponen yang sudah dibuat
            self.model = YOLOv5Model(
                backbone=self.backbone, neck=self.neck, head=self.head,
                detection_layers=self.detection_layers, layer_mode=self.layer_mode,
                loss_fn=self.loss_fn, config=self.config, logger=self.logger, testing_mode=self.testing_mode
            )
            
            # Load pretrained weights jika tersedia
            not self.testing_mode and self.config.get('pretrained', True) and self._load_pretrained_weights()
            
            self.logger.success(f"✅ Model berhasil dibangun: {self.config['backbone']} | {self.layer_mode} | {self.detection_layers}")
            return self.model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun model: {str(e)}")
            raise ModelError(f"Gagal membangun model: {str(e)}")
    
    def _build_backbone(self):
        """Build backbone dengan validation dan error handling"""
        backbone_config = SUPPORTED_BACKBONES.get(self.config['backbone'])
        if not backbone_config: raise ModelError(f"❌ Backbone '{self.config['backbone']}' tidak didukung")
        
        # Build backbone berdasarkan type dengan one-liner
        backbone_builders = {
            'efficientnet': lambda: EfficientNetBackbone(
                model_name=backbone_config['variant'], pretrained=self.config['pretrained'],
                use_attention=self.config.get('use_attention', False), 
                testing_mode=self.testing_mode, logger=self.logger
            ),
            'cspdarknet': lambda: CSPDarknet(
                pretrained=self.config['pretrained'], model_size=backbone_config['variant'],
                testing_mode=self.testing_mode, logger=self.logger
            )
        }
        
        self.backbone = backbone_builders.get(backbone_config['type'], lambda: None)()
        if not self.backbone: raise ModelError(f"❌ Tipe backbone '{backbone_config['type']}' tidak diimplementasikan")
        self.logger.debug(f"✓ Backbone {self.config['backbone']} berhasil diinisialisasi")
    
    def _build_neck(self):
        """Build neck dengan output channels dari backbone"""
        self.backbone or self._build_backbone()
        self.neck = FeatureProcessingNeck(in_channels=self.backbone.get_output_channels(), logger=self.logger)
        self.logger.debug("✓ Feature Neck berhasil diinisialisasi")
    
    def _build_head(self):
        """Build detection head dengan layer configuration"""
        num_classes = get_num_classes_for_layers(self.detection_layers) if self.layer_mode == 'multilayer' else self.config['num_classes']
        self.head = DetectionHead(
            in_channels=self.neck.out_channels, num_classes=num_classes, img_size=self.config['img_size'],
            use_attention=self.config.get('use_attention', False), layer_mode=self.layer_mode,
            detection_layers=self.detection_layers, logger=self.logger
        )
        self.logger.debug(f"✓ Detection head berhasil dibuat: {num_classes} kelas | {self.layer_mode}")
    
    def _build_loss_function(self):
        """Build loss function dengan configuration"""
        self.loss_fn = YOLOLoss(
            num_classes=self.config['num_classes'], use_ciou=self.config.get('use_ciou', False), logger=self.logger
        )
        self.logger.debug(f"✓ Loss function berhasil diinisialisasi: {'CIoU' if self.config.get('use_ciou') else 'IoU'}")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights dari Drive atau default"""
        drive_model_path = check_pretrained_model_in_drive(self.config['backbone'], self.pretrained_models_path)
        
        if drive_model_path:
            self.logger.info(f"🔄 Menggunakan pretrained model dari Drive: {Path(drive_model_path).name}")
            load_pretrained_model(self.model, drive_model_path, self.config.get('device', 'cpu'))
        else:
            self.logger.info("🔄 Menggunakan default pretrained weights")
            load_pretrained_model(self.model, self.config['backbone'], self.config.get('device', 'cpu'))
    
    def get_training_service(self, callback=None):
        """Get training service dengan checkpoint service integration"""
        from smartcash.model.service.training_service import TrainingService
        from smartcash.model.service.checkpoint_service import CheckpointService
        
        # Create checkpoint service jika belum ada
        self.checkpoint_service = self.checkpoint_service or CheckpointService(
            checkpoint_dir=getattr(self, 'save_dir', 'runs/train/checkpoints'), logger=self.logger
        )
        
        training_service = TrainingService(
            model_manager=self, checkpoint_service=self.checkpoint_service, logger=self.logger, callback=callback
        )
        
        self.logger.info(f"✨ Training service dibuat untuk {self.model_type}")
        return training_service
    
    def get_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 0.0005):
        """Create optimizer dengan parameter groups untuk backbone dan head"""
        self.model or self.build_model()
        
        # Parameter groups dengan learning rate berbeda
        param_groups = [
            {'params': list(self.backbone.parameters()), 'lr': learning_rate * 0.1},  # Backbone: LR lebih kecil
            {'params': list(self.head.parameters()), 'lr': learning_rate}             # Head: LR normal
        ]
        return torch.optim.Adam(param_groups, lr=learning_rate, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer, epochs: int = 100):
        """Create learning rate scheduler"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    def save_model(self, path: str) -> str:
        """Save model dengan checkpoint service atau manual"""
        if self.checkpoint_service: return self.checkpoint_service.save_checkpoint(self.model, path)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"✅ Model disimpan ke {path}")
        return path
    
    def load_model(self, path: str) -> nn.Module:
        """Load model dari checkpoint service atau manual"""
        if self.checkpoint_service:
            self.model = self.checkpoint_service.load_checkpoint(path, self.model)[0]
            return self.model
        
        self.model or self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.config['device']))
        self.logger.info(f"✅ Model berhasil diload dari {path}")
        return self.model
    
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Predict dengan preprocessing dan postprocessing"""
        self.model or self.build_model()
        
        # Preprocess input dengan one-liner type checking
        if isinstance(image, str): image = torch.from_numpy(__import__('numpy').array(__import__('PIL.Image', fromlist=['Image']).Image.open(image).convert('RGB'))).permute(2, 0, 1).float() / 255.0
        if isinstance(image, __import__('numpy').ndarray): image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image.dim() == 3 and (image := image.unsqueeze(0))  # Add batch dimension
        
        # Inference dengan device handling
        device = next(self.model.parameters()).device
        with torch.no_grad(): predictions = self.model(image.to(device))
        
        # Postprocess predictions dengan confidence filtering
        results = {}
        for layer_name, layer_preds in predictions.items():
            detections = []
            for pred in layer_preds:
                bs, na, h, w, no = pred.shape
                pred = pred.view(bs, -1, no)
                conf_mask = pred[..., 4] > conf_threshold
                [detections.append(pred[b][conf_mask[b]]) for b in range(bs) if len(pred[b][conf_mask[b]])]
            results[layer_name] = detections
        return results
    
    @classmethod
    def create_model(cls, model_type: str, layer_mode: str = 'single', detection_layers: Optional[List[str]] = None, **kwargs):
        """Factory method untuk create model dengan build otomatis"""
        manager = cls(model_type=model_type, layer_mode=layer_mode, detection_layers=detection_layers, **kwargs)
        manager.build_model()
        return manager
    
    def update_config(self, config_updates: Dict):
        """Update config dengan validation dan rollback protection"""
        old_config = self.config.copy()
        try:
            self.config.update(config_updates)
            # Re-validate layer parameters jika ada perubahan
            if any(key in config_updates for key in ['layer_mode', 'detection_layers']):
                self.layer_mode, self.detection_layers = validate_layer_params(
                    self.config.get('layer_mode', 'single'), self.config.get('detection_layers', ['banknote'])
                )
                self.config.update({'layer_mode': self.layer_mode, 'detection_layers': self.detection_layers})
            self.logger.info("✅ Konfigurasi berhasil diupdate")
        except Exception as e:
            self.config = old_config  # Rollback
            self.logger.error(f"❌ Gagal update konfigurasi: {str(e)}")
            raise ModelConfigurationError(f"Gagal update konfigurasi: {str(e)}")
    
    # One-liner properties dan utilities
    is_model_built = lambda self: self.model is not None
    get_model_type = lambda self: self.model_type
    get_device = lambda self: self.config['device']
    get_model_description = lambda self: OPTIMIZED_MODELS[self.model_type]['description']
    get_backbone_type = lambda self: self.config['backbone']
    get_num_classes = lambda self: self.config['num_classes']
    get_image_size = lambda self: self.config['img_size']
    is_using_attention = lambda self: self.config.get('use_attention', False)
    is_using_residual = lambda self: self.config.get('use_residual', False)
    is_using_ciou = lambda self: self.config.get('use_ciou', False)
    get_layer_mode = lambda self: self.layer_mode
    get_detection_layers = lambda self: self.detection_layers
    is_multilayer = lambda self: self.layer_mode == 'multilayer'
    set_checkpoint_service = lambda self, service: setattr(self, 'checkpoint_service', service) or self
    get_config = lambda self: self.config.copy()