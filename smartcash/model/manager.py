"""
File: smartcash/model/manager.py
Deskripsi: Manajer model yang mengkoordinasikan berbagai arsitektur dan layanan model
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import os
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.model.exceptions import ModelError, ModelConfigurationError

# Imports dari architectures
from smartcash.model.architectures.backbones import (
    EfficientNetBackbone, 
    CSPDarknet
)
from smartcash.model.architectures.necks import FeatureProcessingNeck
from smartcash.model.architectures.heads import DetectionHead

# Imports dari components
from smartcash.model.components import YOLOLoss

class ModelManager:
    """
    Model Manager yang mengkoordinasikan berbagai arsitektur dan layanan model.
    
    Bertanggung jawab untuk:
    - Inisialisasi model dengan konfigurasi yang tepat
    - Integrasi dengan layanan checkpoint, training, dan inference
    - Validasi kompatibilitas antar komponen
    - Manajemen konfigurasi model
    """
    
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
    
    # Konfigurasi deteksi layers
    DETECTION_LAYERS = {
        'banknote': {
            'num_classes': 7,  # 7 denominasi ('001', '002', '005', '010', '020', '050', '100')
            'description': 'Deteksi uang kertas utuh'
        },
        'nominal': {
            'num_classes': 7,  # 7 area nominal ('l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100')
            'description': 'Deteksi area nominal'
        },
        'security': {
            'num_classes': 3,  # 3 fitur keamanan ('l3_sign', 'l3_text', 'l3_thread')
            'description': 'Deteksi fitur keamanan'
        },
        'all': {
            'num_classes': 17,  # semua kelas (7+7+3)
            'description': 'Semua layer deteksi'
        }
    }
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi Model Manager.
        
        Args:
            config: Konfigurasi model (opsional)
            logger: Logger untuk mencatat proses (opsional)
            
        Raises:
            ModelConfigurationError: Jika konfigurasi tidak valid
        """
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
            'anchors': None  # Use default
        }
        
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
        self.training_service = None
        self.evaluation_service = None
        
        self.logger.info(f"✅ ModelManager diinisialisasi dengan konfigurasi:")
        for key, value in self.config.items():
            self.logger.info(f"   • {key}: {value}")
            
    def _validate_config(self):
        """
        Validasi konfigurasi model.
        
        Raises:
            ModelConfigurationError: Jika konfigurasi tidak valid
        """
        # Validasi backbone
        if self.config['backbone'] not in self.SUPPORTED_BACKBONES:
            supported = list(self.SUPPORTED_BACKBONES.keys())
            raise ModelConfigurationError(
                f"❌ Backbone '{self.config['backbone']}' tidak didukung. "
                f"Backbone yang didukung: {supported}"
            )
            
        # Validasi detection layers
        for layer in self.config['detection_layers']:
            if layer not in self.DETECTION_LAYERS and layer != 'all':
                supported = list(self.DETECTION_LAYERS.keys())
                raise ModelConfigurationError(
                    f"❌ Detection layer '{layer}' tidak didukung. "
                    f"Layer yang didukung: {supported}"
                )
                
        # Jika layer 'all', ganti dengan semua layer individual
        if 'all' in self.config['detection_layers']:
            self.config['detection_layers'] = [l for l in self.DETECTION_LAYERS.keys() if l != 'all']
            
        # Validasi num_classes berdasarkan detection layers
        total_classes = sum(self.DETECTION_LAYERS[layer]['num_classes'] 
                           for layer in self.config['detection_layers'])
        if self.config['num_classes'] != total_classes:
            self.logger.warning(
                f"⚠️ Jumlah kelas ({self.config['num_classes']}) tidak sesuai dengan "
                f"total kelas dari detection layers ({total_classes}). "
                f"Menggunakan {total_classes} kelas."
            )
            self.config['num_classes'] = total_classes
            
        # Validasi image size
        if not isinstance(self.config['img_size'], tuple) or len(self.config['img_size']) != 2:
            raise ModelConfigurationError(
                f"❌ Format img_size tidak valid. Harus berupa tuple (width, height)."
            )
                
    def build_model(self) -> nn.Module:
        """
        Buat dan inisialisasi model berdasarkan konfigurasi.
        
        Returns:
            nn.Module: Model yang telah diinisialisasi
            
        Raises:
            ModelError: Jika gagal membangun model
        """
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
            
            self.logger.success(
                f"✅ Model berhasil dibangun dan dipindahkan ke device {self.config['device']}"
            )
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun model: {str(e)}")
            raise ModelError(f"Gagal membangun model: {str(e)}")
            
    def _build_backbone(self):
        """
        Buat backbone berdasarkan konfigurasi.
        
        Raises:
            ModelError: Jika gagal membangun backbone
        """
        try:
            backbone_config = self.SUPPORTED_BACKBONES[self.config['backbone']]
            
            if backbone_config['type'] == 'efficientnet':
                self.backbone = EfficientNetBackbone(
                    pretrained=self.config['pretrained'],
                    model_name=backbone_config['variant'],
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
        """
        Buat neck berdasarkan konfigurasi.
        
        Raises:
            ModelError: Jika gagal membangun neck
        """
        try:
            if not self.backbone:
                raise ModelError("❌ Backbone harus diinisialisasi sebelum neck")
                
            # Dapatkan output channels dari backbone
            in_channels = self.backbone.get_output_channels()
            
            # Standard output channels untuk YOLOv5
            out_channels = [128, 256, 512]
            
            self.neck = FeatureProcessingNeck(
                in_channels=in_channels,
                out_channels=out_channels,
                logger=self.logger
            )
            
            self.logger.info(f"✓ Feature Neck berhasil diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun neck: {str(e)}")
            raise ModelError(f"Gagal membangun neck: {str(e)}")
            
    def _build_head(self):
        """
        Buat detection head berdasarkan konfigurasi.
        
        Raises:
            ModelError: Jika gagal membangun head
        """
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
        """
        Buat loss function berdasarkan konfigurasi.
        
        Raises:
            ModelError: Jika gagal membangun loss function
        """
        try:
            # Buat loss function untuk setiap detection layer
            self.loss_fn = {}
            for layer in self.config['detection_layers']:
                num_classes = self.DETECTION_LAYERS[layer]['num_classes']
                self.loss_fn[layer] = YOLOLoss(
                    num_classes=num_classes,
                    anchors=self.config['anchors'],
                    logger=self.logger
                )
                
            self.logger.info(f"✓ Loss functions berhasil diinisialisasi")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun loss function: {str(e)}")
            raise ModelError(f"Gagal membangun loss function: {str(e)}")
            
    def get_config(self) -> Dict:
        """
        Dapatkan konfigurasi model.
        
        Returns:
            Dict: Konfigurasi model
        """
        return self.config.copy()
        
    def update_config(self, config_updates: Dict) -> Dict:
        """
        Update konfigurasi model.
        
        Args:
            config_updates: Dictionary dengan nilai-nilai konfigurasi baru
            
        Returns:
            Dict: Konfigurasi model yang telah diupdate
            
        Raises:
            ModelConfigurationError: Jika konfigurasi baru tidak valid
        """
        # Simpan konfigurasi lama untuk rollback
        old_config = self.config.copy()
        
        try:
            # Update konfigurasi
            self.config.update(config_updates)
            
            # Validasi konfigurasi baru
            self._validate_config()
            
            self.logger.info(f"✅ Konfigurasi model berhasil diupdate")
            return self.config
            
        except Exception as e:
            # Rollback ke konfigurasi lama
            self.config = old_config
            self.logger.error(f"❌ Gagal update konfigurasi: {str(e)}")
            raise ModelConfigurationError(f"Gagal update konfigurasi: {str(e)}")
    
    def set_checkpoint_service(self, service):
        """Set checkpoint service"""
        self.checkpoint_service = service
        
    def set_training_service(self, service):
        """Set training service"""
        self.training_service = service
        
    def set_evaluation_service(self, service):
        """Set evaluation service"""
        self.evaluation_service = service
        
    def save_model(self, path: str) -> str:
        """
        Simpan model ke file.
        
        Args:
            path: Path untuk menyimpan model
            
        Returns:
            str: Path lengkap file model tersimpan
            
        Raises:
            ModelError: Jika gagal menyimpan model
        """
        if self.checkpoint_service:
            return self.checkpoint_service.save_checkpoint(self.model, path)
        else:
            try:
                # Fallback jika tidak ada checkpoint service
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.model.state_dict(), path)
                self.logger.info(f"✅ Model berhasil disimpan ke {path}")
                return path
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan model: {str(e)}")
                raise ModelError(f"Gagal menyimpan model: {str(e)}")
                
    def load_model(self, path: str) -> nn.Module:
        """
        Load model dari file.
        
        Args:
            path: Path file model
            
        Returns:
            nn.Module: Model yang telah diload
            
        Raises:
            ModelError: Jika gagal load model
        """
        if self.checkpoint_service:
            self.model = self.checkpoint_service.load_checkpoint(path)
            return self.model
        else:
            try:
                # Fallback jika tidak ada checkpoint service
                if not self.model:
                    self.build_model()
                    
                self.model.load_state_dict(torch.load(path, map_location=self.config['device']))
                self.logger.info(f"✅ Model berhasil diload dari {path}")
                return self.model
            except Exception as e:
                self.logger.error(f"❌ Gagal load model: {str(e)}")
                raise ModelError(f"Gagal load model: {str(e)}")
                
    def train(self, *args, **kwargs):
        """
        Latih model.
        
        Raises:
            ModelError: Jika training service tidak tersedia
        """
        if self.training_service:
            return self.training_service.train(self.model, self.loss_fn, *args, **kwargs)
        else:
            raise ModelError("❌ Training service tidak tersedia")
            
    def evaluate(self, *args, **kwargs):
        """
        Evaluasi model.
        
        Raises:
            ModelError: Jika evaluation service tidak tersedia
        """
        if self.evaluation_service:
            return self.evaluation_service.evaluate(self.model, *args, **kwargs)
        else:
            raise ModelError("❌ Evaluation service tidak tersedia")
            
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Lakukan prediksi pada gambar.
        
        Args:
            image: Input image (dapat berupa tensor, numpy array, atau file path)
            conf_threshold: Threshold confidence
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            Hasil prediksi
            
        Raises:
            ModelError: Jika gagal melakukan prediksi
        """
        try:
            if not self.model:
                raise ModelError("❌ Model belum diinisialisasi")
                
            # Preprocess image
            if isinstance(image, str):
                # Load from file path
                from PIL import Image
                import numpy as np
                image = np.array(Image.open(image).convert('RGB'))
                
            if isinstance(image, np.ndarray):
                # Convert numpy array to tensor
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
                
            # Process predictions
            results = {}
            for layer_name, layer_preds in predictions.items():
                # Apply NMS if needed - simplified version
                detections = []
                for pred in layer_preds:
                    # Flatten predictions to [batch, objects, data]
                    bs, na, h, w, no = pred.shape
                    pred = pred.view(bs, -1, no)
                    
                    # Filter by confidence
                    conf_mask = pred[..., 4] > conf_threshold
                    for b in range(bs):
                        # Get detections for this image
                        det = pred[b][conf_mask[b]]
                        if len(det):
                            detections.append(det)
                            
                # Store results
                results[layer_name] = detections
                
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan prediksi: {str(e)}")
            raise ModelError(f"Gagal melakukan prediksi: {str(e)}")


class YOLOv5Model(nn.Module):
    """
    Model YOLOv5 terintegrasi yang menggabungkan backbone, neck, dan head.
    """
    
    def __init__(
        self,
        backbone,
        neck,
        head,
        config: Dict
    ):
        """
        Inisialisasi YOLOv5Model.
        
        Args:
            backbone: Backbone network
            neck: Feature processing neck
            head: Detection head
            config: Konfigurasi model
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.config = config
        
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass model.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dict berisi prediksi untuk setiap layer deteksi
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Process features through neck
        processed_features = self.neck(features)
        
        # Get predictions from head
        predictions = self.head(processed_features)
        
        return predictions
    
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45
    ) -> Dict[str, List[Dict]]:
        """
        Lakukan prediksi dengan post-processing.
        
        Args:
            x: Input tensor
            conf_threshold: Threshold confidence
            nms_threshold: Threshold IoU untuk NMS
            
        Returns:
            Dict berisi hasil deteksi untuk setiap layer
        """
        # Get raw predictions
        predictions = self(x)
        
        # Process predictions for each layer
        results = {}
        
        for layer_name, layer_preds in predictions.items():
            # TODO: Implement proper NMS and post-processing
            results[layer_name] = [{"raw_predictions": p} for p in layer_preds]
            
        return results