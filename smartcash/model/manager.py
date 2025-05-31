"""
File: smartcash/model/manager.py
Deskripsi: Updated model manager dengan simplified training integration
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import os
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from smartcash.common.logger import SmartCashLogger
from smartcash.model.utils.pretrained_model_utils import check_pretrained_model_in_drive, load_pretrained_model
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

# Import konstanta dari config
from smartcash.model.config.model_constants import SUPPORTED_BACKBONES, OPTIMIZED_MODELS, get_model_config, DETECTION_LAYERS
from smartcash.model.config import DEFAULT_MODEL_CONFIG

# Import validator layer
from smartcash.model.utils.layer_validator import validate_layer_params, get_num_classes_for_layers

class ModelManager:
    """Model Manager dengan simplified training integration untuk YOLOv5 + EfficientNet-B4"""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        model_type: str = 'efficient_optimized',
        layer_mode: str = 'single',
        detection_layers: Optional[List[str]] = None,
        testing_mode: bool = False,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi Model Manager dengan simplified configuration"""
        self.logger = logger or SmartCashLogger(__name__)
        
        # Default konfigurasi dari konstanta
        self.default_config = DEFAULT_MODEL_CONFIG.copy()
        
        # Verifikasi tipe model
        if model_type not in OPTIMIZED_MODELS:
            self.logger.warning(f"⚠️ Tipe model '{model_type}' tidak dikenal, menggunakan 'efficient_optimized'")
            model_type = 'efficient_optimized'
            
        self.model_type = model_type
        
        # Update default config dengan konfigurasi tipe model
        model_config = OPTIMIZED_MODELS[model_type]
        for key, value in model_config.items():
            if key != 'description':
                self.default_config[key] = value
        
        # Simpan config dan validasi
        self.config = self.default_config.copy()
        self.config.update(config or {})
        
        # Gunakan nilai dari config jika parameter langsung tidak diberikan
        layer_mode = layer_mode if layer_mode != 'single' else self.config.get('layer_mode', 'single')
        detection_layers = detection_layers or self.config.get('detection_layers', ['banknote'])
        
        # Debug logging untuk layer_mode dan detection_layers
        self.logger.info(f"📝 ModelManager init: layer_mode={layer_mode}, detection_layers={detection_layers}")
        
        # PENTING: Jika mode multilayer dengan multiple layers, JANGAN ubah ke single
        if layer_mode == 'multilayer' and len([l for l in detection_layers if l in DETECTION_LAYERS]) >= 2:
            # Jika sudah multilayer dengan 2+ layer valid, pertahankan mode ini
            valid_layer_mode = 'multilayer'
            valid_detection_layers = [l for l in detection_layers if l in DETECTION_LAYERS]
            if len(valid_detection_layers) != len(detection_layers):
                self.logger.warning(f"⚠️ Beberapa detection_layers tidak valid, hanya menggunakan yang valid: {valid_detection_layers}")
        else:
            # Gunakan validasi standar untuk kasus lainnya
            valid_layer_mode, valid_detection_layers = validate_layer_params(layer_mode, detection_layers)
            
        # Debug logging setelah validasi
        self.logger.info(f"📝 ModelManager setelah validasi: layer_mode={valid_layer_mode}, detection_layers={valid_detection_layers}")
        
        # Update config dengan parameter layer yang sudah divalidasi
        self.config['layer_mode'] = valid_layer_mode
        self.config['detection_layers'] = valid_detection_layers
        
        # Update num_classes berdasarkan detection_layers jika mode multilayer
        if valid_layer_mode == 'multilayer':
            self.config['num_classes'] = get_num_classes_for_layers(valid_detection_layers)
        
        self.testing_mode = testing_mode
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
        self.logger.info(f"   • Layer Mode: {self.config['layer_mode']}")
        self.logger.info(f"   • Detection Layers: {self.config['detection_layers']}")
        for key, value in self.config.items():
            if key not in ['layer_mode', 'detection_layers']:
                self.logger.info(f"   • {key}: {value}")
            
    def _validate_config(self):
        """Validasi konfigurasi model"""
        # Validasi backbone
        if self.config['backbone'] not in SUPPORTED_BACKBONES:
            supported = list(SUPPORTED_BACKBONES.keys())
            raise ModelConfigurationError(f"❌ Backbone '{self.config['backbone']}' tidak didukung. Backbone yang didukung: {supported}")
            
        # Dapatkan layer_config
        layer_config = get_layer_config()
        all_layers = layer_config.get_layer_names()
            
        # Validasi detection layers
        for layer in self.config['detection_layers']:
            if layer not in all_layers and layer != 'all':
                supported = list(all_layers)
                raise ModelConfigurationError(f"❌ Detection layer '{layer}' tidak didukung. Layer yang didukung: {supported}")
                
        # Validasi layer_mode
        if self.config.get('layer_mode') not in ['single', 'multilayer']:
            self.logger.warning(f"⚠️ Layer mode '{self.config.get('layer_mode')}' tidak valid, menggunakan 'single'")
            self.config['layer_mode'] = 'single'
            
        # Validasi konsistensi layer_mode dan detection_layers
        valid_layer_mode, valid_detection_layers = validate_layer_params(
            self.config.get('layer_mode', 'single'),
            self.config.get('detection_layers', ['banknote'])
        )
        
        # Update config dengan parameter layer yang sudah divalidasi
        self.config['layer_mode'] = valid_layer_mode
        self.config['detection_layers'] = valid_detection_layers
                
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
    
    # Metode _check_pretrained_model_in_drive telah dipindahkan ke pretrained_model_utils.py
    
    def build_model(self) -> nn.Module:
        """Buat dan inisialisasi model berdasarkan konfigurasi"""
        try:
            # Buat komponen model
            self._build_backbone()
            self._build_neck()
            self._build_head()
            self._build_loss_function()
            
            # Gunakan layer_mode dan detection_layers dari head yang sudah dibuat
            # untuk memastikan konsistensi
            head_layer_mode = getattr(self.head, 'layer_mode', self.config.get('layer_mode', 'single'))
            head_detection_layers = getattr(self.head, 'detection_layers', self.config.get('detection_layers', ['banknote']))
            
            # Buat model YOLOv5 dengan komponen yang sudah dibuat
            self.model = YOLOv5Model(
                backbone=self.backbone,
                neck=self.neck,
                head=self.head,
                detection_layers=head_detection_layers,
                layer_mode=head_layer_mode
            )
            
            # Cek apakah ada pretrained model di drive dan load jika ada
            if self.config.get('pretrained', True) and not self.testing_mode:
                drive_model_path = check_pretrained_model_in_drive(self.config['backbone'])
                if drive_model_path:
                    self.logger.info(f"🔄 Menggunakan pretrained model dari Google Drive: {drive_model_path}")
                    self.load_model(drive_model_path)
                    return self.model
                
                # Jika tidak ada di drive, load dari pretrained default
                self.logger.info(f"🔄 Tidak ada pretrained model di Google Drive, menggunakan default pretrained")
                load_pretrained_model(self.model, self.config['backbone'])
            
            self.logger.info(f"✅ Model berhasil dibangun dengan backbone {self.config['backbone']}")
            self.logger.info(f"   • Layer Mode: {self.config['layer_mode']}")
            self.logger.info(f"   • Detection Layers: {self.config['detection_layers']}")
            return self.model
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun model: {str(e)}")
            raise ModelError(f"Gagal membangun model: {str(e)}")
            
    def _build_backbone(self):
        """Buat backbone berdasarkan konfigurasi"""
        try:
            backbone_config = SUPPORTED_BACKBONES.get(self.config['backbone'], None)
            if not backbone_config: raise ModelError(f"❌ Backbone '{self.config['backbone']}' tidak didukung")
            
            # Buat backbone berdasarkan tipe
            backbone_builders = {
                'efficientnet': lambda: EfficientNetBackbone(pretrained=self.config['pretrained'], model_name=backbone_config['variant'], use_attention=self.config.get('use_attention', False), testing_mode=self.testing_mode, logger=self.logger),
                'cspdarknet': lambda: CSPDarknet(pretrained=self.config['pretrained'], model_size=backbone_config['variant'], testing_mode=self.testing_mode, logger=self.logger)
            }
            
            # Gunakan builder yang sesuai atau raise error jika tidak ada
            self.backbone = backbone_builders.get(backbone_config['type'], lambda: None)()
            if not self.backbone: raise ModelError(f"❌ Tipe backbone '{backbone_config['type']}' tidak diimplementasikan")
            
            self.logger.info(f"✓ Backbone {self.config['backbone']} berhasil diinisialisasi")
            return self.backbone
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun backbone: {str(e)}")
            raise ModelError(f"Gagal membangun backbone: {str(e)}")
            
    def _build_neck(self):
        """Buat neck berdasarkan konfigurasi"""
        try:
            # Pastikan backbone sudah dibangun dan dapatkan output channels
            if not self.backbone: self._build_backbone()
            output_channels = self.backbone.get_output_channels()
            
            # Buat neck dengan FeatureProcessingNeck
            self.neck = FeatureProcessingNeck(in_channels=output_channels, logger=self.logger)
            feature_type = "standar"
            self.logger.info(f"✓ Feature Neck {feature_type} berhasil diinisialisasi")
            return self.neck
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun neck: {str(e)}")
            raise ModelError(f"Gagal membangun neck: {str(e)}")
            
    def _build_head(self):
        """Buat detection head berdasarkan konfigurasi"""
        try:
            # Dapatkan jumlah kelas dan ukuran gambar
            num_classes = self.config['num_classes']
            img_size = self.config['img_size']
            layer_mode = self.config.get('layer_mode', 'single')
            detection_layers = self.config.get('detection_layers', ['banknote'])
            
            # Debug logging untuk layer_mode dan detection_layers
            self.logger.info(f"🔍 Membangun detection head dengan layer_mode: {layer_mode}, detection_layers: {detection_layers}")
            
            # Jika mode multilayer, hitung ulang num_classes berdasarkan detection_layers
            if layer_mode == 'multilayer':
                num_classes = get_num_classes_for_layers(detection_layers)
                self.config['num_classes'] = num_classes
                self.logger.info(f"ℹ️ Mode multilayer: menggunakan {num_classes} kelas total dari {len(detection_layers)} layers")
            
            # Validasi layer_mode dan detection_layers untuk multilayer
            if layer_mode == 'multilayer' and len(detection_layers) < 2:
                self.logger.warning(f"⚠️ Mode multilayer membutuhkan minimal 2 detection_layers, tetapi hanya {len(detection_layers)} yang diberikan.")
            
            # Buat detection head
            self.head = DetectionHead(
                in_channels=self.neck.out_channels,
                num_classes=num_classes,
                img_size=img_size,
                use_attention=self.config.get('use_attention', False),
                layer_mode=layer_mode,
                detection_layers=detection_layers
            )
            
            # Debug logging untuk layer_mode setelah membuat head
            self.logger.info(f"🔍 Detection head dibuat dengan layer_mode: {self.head.layer_mode}, detection_layers: {self.head.detection_layers}")
            
            self.logger.info(f"✅ Detection head berhasil dibuat dengan {num_classes} kelas")
            return self.head
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat detection head: {str(e)}")
            raise ModelError(f"Gagal membuat detection head: {str(e)}")
            
    def _build_loss_function(self):
        """Buat loss function berdasarkan konfigurasi"""
        try:
            # Buat loss function dengan YOLOLoss
            self.loss_fn = YOLOLoss(num_classes=self.config['num_classes'], use_ciou=self.config.get('use_ciou', False), logger=self.logger)
            loss_type = "CIoU" if self.config.get('use_ciou', False) else "standar"
            self.logger.info(f"✓ Loss function ({loss_type}) berhasil diinisialisasi")
            return self.loss_fn
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun loss function: {str(e)}")
            raise ModelError(f"Gagal membangun loss function: {str(e)}")
    
    def get_training_service(self, callback=None):
        """Dapatkan training service yang terintegrasi dengan model manager ini
        
        Args:
            callback: Callback untuk progress tracking dan metrics reporting
            
        Returns:
            TrainingService: Instance training service
        """
        from smartcash.model.service.training_service import TrainingService
        from smartcash.model.service.checkpoint_service import CheckpointService
        
        # Buat checkpoint service jika belum ada
        if not hasattr(self, 'checkpoint_service') or self.checkpoint_service is None:
            save_dir = getattr(self, 'save_dir', 'runs/train')
            self.checkpoint_service = CheckpointService(
                checkpoint_dir=os.path.join(save_dir, "checkpoints"),
                logger=self.logger
            )
            
        # Buat training service
        training_service = TrainingService(
            model_manager=self,
            checkpoint_service=self.checkpoint_service,
            logger=self.logger,
            callback=callback
        )
        
        self.logger.info(f"✨ Training service dibuat untuk model {self.model_type}")
        return training_service
    
    def get_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 0.0005):
        """Buat optimizer dengan parameter groups untuk backbone dan head"""
        if not self.model: self.build_model()
        
        # Buat parameter groups dengan learning rate berbeda untuk backbone dan head
        param_groups = [
            {'params': list(self.backbone.parameters()), 'lr': learning_rate * 0.1},  # Backbone: learning rate lebih kecil
            {'params': list(self.head.parameters()), 'lr': learning_rate}             # Head: learning rate normal
        ]
        
        return torch.optim.Adam(param_groups, lr=learning_rate, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer, epochs: int = 100):
        """Buat learning rate scheduler"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    @classmethod
    def create_model(cls, model_type: str, layer_mode: str = 'single', detection_layers: Optional[List[str]] = None, **kwargs):
        """Factory method untuk membuat model berdasarkan tipe"""
        manager = cls(model_type=model_type, layer_mode=layer_mode, detection_layers=detection_layers, **kwargs)
        manager.build_model()
        return manager
        
    def get_config(self) -> Dict:
        """Dapatkan konfigurasi model saat ini"""
        return self.config.copy()
        
    def get_layer_mode(self) -> str:
        """Dapatkan mode layer (single atau multilayer)"""
        if hasattr(self, 'model') and self.model is not None:
            return self.model.layer_mode
        elif hasattr(self, 'head') and self.head is not None:
            return self.head.layer_mode
        return self.config.get('layer_mode', 'single')
    
    def get_detection_layers(self) -> List[str]:
        """Dapatkan daftar detection layers yang digunakan"""
        if hasattr(self, 'model') and self.model is not None:
            return self.model.detection_layers
        return self.config.get('detection_layers', ['banknote'])
    
    def is_multilayer(self) -> bool:
        """Cek apakah model menggunakan mode multilayer"""
        return self.get_layer_mode() == 'multilayer'
        
    def update_config(self, config_updates: Dict):
        """Update konfigurasi model"""
        try:
            old_config = self.config.copy()  # Simpan konfigurasi lama untuk rollback
            self.config.update(config_updates)  # Update konfigurasi
            self._validate_config()  # Validasi konfigurasi baru
            self.logger.info(f"✅ Konfigurasi berhasil diupdate")
        except Exception as e:
            self.config = old_config  # Rollback jika terjadi error
            self.logger.error(f"❌ Gagal update konfigurasi: {str(e)}")
            raise ModelConfigurationError(f"Gagal update konfigurasi: {str(e)}")
    
    def set_checkpoint_service(self, checkpoint_service):
        """Set checkpoint service untuk model manager"""
        self.checkpoint_service = checkpoint_service
        self.logger.info(f"Checkpoint service diatur: {type(checkpoint_service).__name__}")
        return self
        
    def save_model(self, path: str) -> str:
        """Simpan model ke file"""
        if self.checkpoint_service: return self.checkpoint_service.save_checkpoint(self.model, path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)  # Buat direktori jika belum ada
            torch.save(self.model.state_dict(), path)  # Simpan model state dict
            self.logger.info(f"✅ Model berhasil disimpan ke {path}")
            return path
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan model: {str(e)}")
            raise ModelError(f"Gagal menyimpan model: {str(e)}")
                
    def load_model(self, path: str) -> nn.Module:
        """Load model dari file"""
        if self.checkpoint_service: 
            self.model = self.checkpoint_service.load_checkpoint(path, self.model)
            return self.model
        try:
            if not self.model: self.build_model()  # Bangun model jika belum ada
            self.model.load_state_dict(torch.load(path, map_location=self.config['device']))  # Load state dict
            self.logger.info(f"✅ Model berhasil diload dari {path}")
            return self.model
        except Exception as e:
            self.logger.error(f"❌ Gagal load model: {str(e)}")
            raise ModelError(f"Gagal load model: {str(e)}")
    
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Lakukan prediksi pada gambar"""
        try:
            if not self.model: raise ModelError("❌ Model belum diinisialisasi")
            
            # Preprocess image berdasarkan tipe input
            if isinstance(image, str): 
                from PIL import Image
                import numpy as np
                image = np.array(Image.open(image).convert('RGB'))
            if isinstance(image, np.ndarray): image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if image.dim() == 3: image = image.unsqueeze(0)  # Tambahkan dimensi batch jika perlu
            
            # Pindahkan ke device yang benar dan lakukan prediksi
            device = next(self.model.parameters()).device
            image = image.to(device)
            with torch.no_grad(): predictions = self.model(image)
            
            # Proses hasil prediksi dengan filter confidence
            results = {}
            for layer_name, layer_preds in predictions.items():
                detections = []
                for pred in layer_preds:
                    bs, na, h, w, no = pred.shape
                    pred = pred.view(bs, -1, no)
                    conf_mask = pred[..., 4] > conf_threshold  # Filter berdasarkan confidence
                    for b in range(bs):
                        det = pred[b][conf_mask[b]]
                        if len(det): detections.append(det)
                results[layer_name] = detections
            return results
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan prediksi: {str(e)}")
            raise ModelError(f"Gagal melakukan prediksi: {str(e)}")
    
    # One-liner utilities
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
    get_layer_mode = lambda self: self.config.get('layer_mode', 'single')
    get_detection_layers = lambda self: self.config.get('detection_layers', ['banknote'])
    is_multilayer = lambda self: self.config.get('layer_mode', 'single') == 'multilayer'