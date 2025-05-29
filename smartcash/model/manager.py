"""
File: smartcash/model/manager.py
Deskripsi: Updated model manager dengan simplified training integration
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import os
from pathlib import Path

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import ModelError, ModelConfigurationError
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.layer_config import get_layer_config

# Imports dari architectures
from smartcash.model.architectures.backbones import EfficientNetBackbone, CSPDarknet
from smartcash.model.architectures.necks import FeatureProcessingNeck
from smartcash.model.architectures.heads import DetectionHead

# Imports dari components
from smartcash.model.components import YOLOLoss

# Import model
from smartcash.model.models.yolov5_model import YOLOv5Model

# Definisi layer deteksi untuk digunakan di seluruh aplikasi
DETECTION_LAYERS = ['banknote', 'nominal', 'security']

class ModelManager:
    """Model Manager dengan simplified training integration untuk YOLOv5 + EfficientNet-B4"""
    
    # Enum untuk backbone yang didukung
    SUPPORTED_BACKBONES = {
        'efficientnet_b0': {'type': 'efficientnet', 'variant': 'efficientnet_b0'},
        'efficientnet_b1': {'type': 'efficientnet', 'variant': 'efficientnet_b1'},
        'efficientnet_b2': {'type': 'efficientnet', 'variant': 'efficientnet_b2'},
        'efficientnet_b3': {'type': 'efficientnet', 'variant': 'efficientnet_b3'},
        'efficientnet_b4': {'type': 'efficientnet', 'variant': 'efficientnet_b4'},
        'efficientnet_b5': {'type': 'efficientnet', 'variant': 'efficientnet_b5'},
        'cspdarknet_s': {'type': 'cspdarknet', 'variant': 'yolov5s'},
        'cspdarknet_m': {'type': 'cspdarknet', 'variant': 'yolov5m'},
        'cspdarknet_l': {'type': 'cspdarknet', 'variant': 'yolov5l'},
    }
    
    # Konfigurasi untuk model teroptimasi
    OPTIMIZED_MODELS = {
        'yolov5s': {
            'description': 'YOLOv5s dengan CSPDarknet sebagai backbone (model pembanding)',
            'backbone': 'cspdarknet_s',
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False
        },
        'efficient_basic': {
            'description': 'Model dasar tanpa optimasi khusus',
            'backbone': 'efficientnet_b4',
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False
        },
        'efficient_optimized': {
            'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
            'backbone': 'efficientnet_b4',
            'use_attention': True,
            'use_residual': False,
            'use_ciou': False
        },
        'efficient_advanced': {
            'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
            'backbone': 'efficientnet_b4',
            'use_attention': True,
            'use_residual': True,
            'use_ciou': True
        },
        'efficient_experiment': {
            'description': 'Model penelitian dengan konfigurasi khusus',
            'backbone': 'efficientnet_b4',
            'use_attention': True,
            'use_residual': True,
            'use_ciou': True,
            'num_repeats': 3
        }
    }
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        model_type: str = 'efficient_optimized',
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi Model Manager dengan simplified configuration"""
        self.logger = logger or SmartCashLogger(__name__)
        
        # Default konfigurasi
        self.default_config = {
            'backbone': 'efficientnet_b4',
            'pretrained': True,
            'detection_layers': ['banknote'],
            'num_classes': 7,
            'img_size': (640, 640),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'checkpoint_dir': 'checkpoints',
            'batch_size': 16,
            'dropout': 0.0,
            'anchors': None,
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False
        }
        
        # Verifikasi tipe model
        if model_type not in self.OPTIMIZED_MODELS:
            self.logger.warning(f"⚠️ Tipe model '{model_type}' tidak dikenal, menggunakan 'efficient_optimized'")
            model_type = 'efficient_optimized'
            
        self.model_type = model_type
        
        # Update default config dengan konfigurasi tipe model
        model_config = self.OPTIMIZED_MODELS[model_type]
        for key, value in model_config.items():
            if key != 'description':
                self.default_config[key] = value
        
        # Merge konfigurasi
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Validasi konfigurasi
        self._validate_config()
        
        # Komponen-komponen model
        self.backbone = None
        self.neck = None
        self.head = None
        self.model = None
        self.loss_fn = None
        
        # Services dependencies (akan diset dari luar)
        self.checkpoint_service = None
        
        self.logger.info(f"✅ ModelManager diinisialisasi dengan model {model_type}:")
        self.logger.info(f"   • {model_config['description']}")
        for key, value in self.config.items():
            self.logger.info(f"   • {key}: {value}")
            
    def _validate_config(self):
        """Validasi konfigurasi model"""
        # Validasi backbone
        if self.config['backbone'] not in self.SUPPORTED_BACKBONES:
            supported = list(self.SUPPORTED_BACKBONES.keys())
            raise ModelConfigurationError(f"❌ Backbone '{self.config['backbone']}' tidak didukung. Backbone yang didukung: {supported}")
            
        # Dapatkan layer_config
        layer_config = get_layer_config()
        all_layers = layer_config.get_layer_names()
            
        # Validasi detection layers
        for layer in self.config['detection_layers']:
            if layer not in all_layers and layer != 'all':
                supported = list(all_layers)
                raise ModelConfigurationError(f"❌ Detection layer '{layer}' tidak didukung. Layer yang didukung: {supported}")
                
        # Jika layer 'all', ganti dengan semua layer individual
        if 'all' in self.config['detection_layers']:
            self.config['detection_layers'] = all_layers
            
        # Validasi num_classes berdasarkan detection layers
        total_classes = 0
        for layer in self.config['detection_layers']:
            layer_config_data = layer_config.get_layer_config(layer)
            total_classes += len(layer_config_data.get('class_ids', []))
            
        if self.config['num_classes'] != total_classes:
            self.logger.warning(f"⚠️ Jumlah kelas ({self.config['num_classes']}) disesuaikan ke {total_classes}")
            self.config['num_classes'] = total_classes
    
    def build_model(self) -> nn.Module:
        """Buat dan inisialisasi model berdasarkan konfigurasi"""
        try:
            # 1. Buat backbone
            self._build_backbone()
            
            # 2. Buat neck
            self._build_neck()
            
            # 3. Buat head
            self._build_head()
            
            # 4. Buat loss function
            self._build_loss_function()
            
            # 5. Integrasi model
            self.model = YOLOv5Model(
                backbone=self.backbone,
                neck=self.neck,
                head=self.head,
                config=self.config
            )
            
            # 6. Pindahkan model ke device yang diinginkan
            device = torch.device(self.config['device'])
            self.model.to(device)
            
            model_desc = self.OPTIMIZED_MODELS[self.model_type]['description']
            self.logger.success(f"✅ Model {self.model_type} ({model_desc}) berhasil dibangun pada device {self.config['device']}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun model: {str(e)}")
            raise ModelError(f"Gagal membangun model: {str(e)}")
            
    def _build_backbone(self):
        """Buat backbone berdasarkan konfigurasi"""
        try:
            backbone_config = self.SUPPORTED_BACKBONES[self.config['backbone']]
            
            if backbone_config['type'] == 'efficientnet':
                self.backbone = EfficientNetBackbone(
                    pretrained=self.config['pretrained'],
                    model_name=backbone_config['variant'],
                    use_attention=self.config.get('use_attention', False),
                    logger=self.logger
                )
            elif backbone_config['type'] == 'cspdarknet':
                self.backbone = CSPDarknet(
                    pretrained=self.config['pretrained'],
                    model_size=backbone_config['variant'],
                    logger=self.logger
                )
            else:
                raise ModelError(f"❌ Tipe backbone '{backbone_config['type']}' tidak diimplementasikan")
                
            self.logger.info(f"✓ Backbone {self.config['backbone']} berhasil diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun backbone: {str(e)}")
            raise ModelError(f"Gagal membangun backbone: {str(e)}")
            
    def _build_neck(self):
        """Buat neck berdasarkan konfigurasi"""
        try:
            if not self.backbone:
                raise ModelError("❌ Backbone harus diinisialisasi sebelum neck")
                
            # Dapatkan output channels dari backbone
            in_channels = self.backbone.get_output_channels()
            
            # Standard output channels untuk YOLOv5
            out_channels = [128, 256, 512]
            
            # Konfigurasi tambahan untuk residual blocks
            num_repeats = self.config.get('num_repeats', 3) if self.config.get('use_residual', False) else 1
            
            self.neck = FeatureProcessingNeck(
                in_channels=in_channels,
                out_channels=out_channels,
                num_repeats=num_repeats,
                logger=self.logger
            )
            
            feature_type = "dengan ResidualAdapter" if self.config.get('use_residual', False) else "standar"
            self.logger.info(f"✓ Feature Neck {feature_type} berhasil diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun neck: {str(e)}")
            raise ModelError(f"Gagal membangun neck: {str(e)}")
            
    def _build_head(self):
        """Buat detection head berdasarkan konfigurasi"""
        try:
            if not self.neck:
                raise ModelError("❌ Neck harus diinisialisasi sebelum head")
                
            # Standard output channels dari neck
            in_channels = [128, 256, 512]
            
            self.head = DetectionHead(
                in_channels=in_channels,
                layers=self.config['detection_layers'],
                logger=self.logger
            )
            
            self.logger.info(f"✓ Detection Head berhasil diinisialisasi untuk layer {self.config['detection_layers']}")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun detection head: {str(e)}")
            raise ModelError(f"Gagal membangun detection head: {str(e)}")
            
    def _build_loss_function(self):
        """Buat loss function berdasarkan konfigurasi"""
        try:
            # Buat loss function untuk setiap detection layer
            self.loss_fn = {}
            layer_config = get_layer_config()
            
            for layer in self.config['detection_layers']:
                layer_data = layer_config.get_layer_config(layer)
                num_classes = len(layer_data.get('class_ids', []))
                self.loss_fn[layer] = YOLOLoss(
                    num_classes=num_classes,
                    anchors=self.config['anchors'],
                    use_ciou=self.config.get('use_ciou', False),
                    logger=self.logger
                )
                
            loss_type = "CIoU" if self.config.get('use_ciou', False) else "IoU"
            self.logger.info(f"✓ Loss functions ({loss_type}) berhasil diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun loss function: {str(e)}")
            raise ModelError(f"Gagal membangun loss function: {str(e)}")
    
    def get_training_service(self, config: Dict[str, Any] = None):
        """Get training service terintegrasi dengan model manager"""
        try:
            from smartcash.model.services.training_service import ModelTrainingService
            return ModelTrainingService(self, config)
        except ImportError as e:
            self.logger.error(f"❌ Training service tidak tersedia: {str(e)}")
            return None
    
    def get_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 0.0005):
        """Buat optimizer dengan parameter groups untuk backbone dan head"""
        if not self.model:
            raise ModelError("❌ Model harus dibangun sebelum membuat optimizer")
        
        # Split parameter untuk grup berbeda
        backbone_params = [p for n, p in self.model.named_parameters() if 'backbone' in n and p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if ('head' in n or 'neck' in n) and p.requires_grad]
        
        # Buat parameter groups dengan different learning rate
        param_groups = [
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Backbone dengan LR lebih rendah
            {'params': head_params, 'lr': learning_rate}             # Head dengan LR normal
        ]
        
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer, epochs: int = 100):
        """Buat learning rate scheduler"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> 'ModelManager':
        """Factory method untuk membuat model berdasarkan tipe"""
        if model_type not in cls.OPTIMIZED_MODELS:
            available_types = list(cls.OPTIMIZED_MODELS.keys())
            raise ModelConfigurationError(f"❌ Tipe model '{model_type}' tidak dikenal. Tipe yang tersedia: {available_types}")
            
        return cls(model_type=model_type, **kwargs)
            
    def get_config(self) -> Dict:
        """Dapatkan konfigurasi model"""
        return self.config.copy()
        
    def update_config(self, config_updates: Dict) -> Dict:
        """Update konfigurasi model"""
        old_config = self.config.copy()
        
        try:
            self.config.update(config_updates)
            self._validate_config()
            self.logger.info(f"✅ Konfigurasi model berhasil diupdate")
            return self.config
            
        except Exception as e:
            self.config = old_config
            self.logger.error(f"❌ Gagal update konfigurasi: {str(e)}")
            raise ModelConfigurationError(f"Gagal update konfigurasi: {str(e)}")
    
    def set_checkpoint_service(self, service: ICheckpointService):
        """Set checkpoint service untuk model manager"""
        self.checkpoint_service = service
        
    def save_model(self, path: str) -> str:
        """Simpan model ke file"""
        if self.checkpoint_service:
            return self.checkpoint_service.save_checkpoint(self.model, path)
        else:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.model.state_dict(), path)
                self.logger.info(f"✅ Model berhasil disimpan ke {path}")
                return path
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan model: {str(e)}")
                raise ModelError(f"Gagal menyimpan model: {str(e)}")
                
    def load_model(self, path: str) -> nn.Module:
        """Load model dari file"""
        if self.checkpoint_service:
            loaded_checkpoint = self.checkpoint_service.load_checkpoint(path, self.model)
            self.model = loaded_checkpoint
            return self.model
        else:
            try:
                if not self.model:
                    self.build_model()
                    
                self.model.load_state_dict(torch.load(path, map_location=self.config['device']))
                self.logger.info(f"✅ Model berhasil diload dari {path}")
                return self.model
            except Exception as e:
                self.logger.error(f"❌ Gagal load model: {str(e)}")
                raise ModelError(f"Gagal load model: {str(e)}")
    
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Lakukan prediksi pada gambar"""
        try:
            if not self.model:
                raise ModelError("❌ Model belum diinisialisasi")
                
            # Preprocess image
            if isinstance(image, str):
                from PIL import Image
                import numpy as np
                image = np.array(Image.open(image).convert('RGB'))
                
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                
            # Ensure batch dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # Move to correct device
            device = next(self.model.parameters()).device
            image = image.to(device)
                
            # Get predictions
            with torch.no_grad():
                predictions = self.model(image)
                
            # Process predictions dengan NMS
            results = {}
            for layer_name, layer_preds in predictions.items():
                detections = []
                for pred in layer_preds:
                    bs, na, h, w, no = pred.shape
                    pred = pred.view(bs, -1, no)
                    
                    # Filter by confidence
                    conf_mask = pred[..., 4] > conf_threshold
                    for b in range(bs):
                        det = pred[b][conf_mask[b]]
                        if len(det):
                            detections.append(det)
                            
                results[layer_name] = detections
                
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan prediksi: {str(e)}")
            raise ModelError(f"Gagal melakukan prediksi: {str(e)}")
    
    # One-liner utilities
    is_model_built = lambda self: self.model is not None
    get_model_type = lambda self: self.model_type
    get_device = lambda self: self.config['device']
    get_model_description = lambda self: self.OPTIMIZED_MODELS[self.model_type]['description']