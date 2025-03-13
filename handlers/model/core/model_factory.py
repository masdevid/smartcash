# File: smartcash/handlers/model/core/model_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk membuat model dengan backbone yang berbeda, direfaktor untuk konsistensi

import torch
from typing import Dict, Optional, Union, List, Tuple

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.model.core.backbone_factory import BackboneFactory 
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.core.model_component import ModelComponent

class ModelFactory(ModelComponent):
    """
    Factory untuk membuat model dengan backbone yang berbeda.
    Direfaktor untuk menggunakan lazy-loading dari ModelComponent.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi model factory.
        
        Args:
            config: Konfigurasi model dan training
            logger: Custom logger (opsional)
        """
        super().__init__(config, logger, "model_factory")
        
    def _initialize(self) -> None:
        """Inisialisasi internal komponen factory."""
        # Setup konfigurasi model dari config
        model_config = self.config.get('model', {})
        self.num_classes = model_config.get('num_classes', 7)
        
        # Dapatkan informasi layer dari config
        self.detection_layers = self.config.get('layers', {})
    
    @property
    def backbone_factory(self):
        """Lazy-loaded backbone factory."""
        return self.get_component('backbone_factory', lambda: BackboneFactory(self.config, self.logger))
    
    def process(
        self, 
        backbone_type: Optional[str] = None,
        pretrained: Optional[bool] = None,
        num_classes: Optional[int] = None,
        detection_layers: Optional[List[str]] = None,
        **kwargs
    ) -> torch.nn.Module:
        """
        Membuat model dengan backbone dan konfigurasi yang ditentukan.
        Alias untuk create_model().
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            pretrained: Gunakan pretrained weights pada backbone
            num_classes: Jumlah kelas untuk deteksi
            detection_layers: Layer deteksi yang diaktifkan
            
        Returns:
            Model yang siap digunakan
        """
        return self.create_model(
            backbone_type=backbone_type,
            pretrained=pretrained,
            num_classes=num_classes,
            detection_layers=detection_layers,
            **kwargs
        )
    
    def create_model(
        self, 
        backbone_type: Optional[str] = None,
        pretrained: Optional[bool] = None,
        num_classes: Optional[int] = None,
        detection_layers: Optional[List[str]] = None,
        **kwargs
    ) -> torch.nn.Module:
        """
        Membuat model dengan backbone dan konfigurasi yang ditentukan.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            pretrained: Gunakan pretrained weights pada backbone
            num_classes: Jumlah kelas untuk deteksi
            detection_layers: Layer deteksi yang diaktifkan
            
        Returns:
            Model yang siap digunakan
        """
        # Prioritaskan parameter yang diberikan, atau gunakan dari config
        backbone_type = backbone_type or self.config.get('model', {}).get('backbone', 'efficientnet')
        pretrained = pretrained if pretrained is not None else self.config.get('model', {}).get('pretrained', True)
        num_classes = num_classes or self.num_classes
        detection_layers = detection_layers or list(self.detection_layers.keys())
        
        # Log informasi model
        self.logger.info(
            f"🔄 Membuat model dengan backbone: {backbone_type}\n"
            f"   • Pretrained: {pretrained}\n"
            f"   • Detection layers: {detection_layers}\n"
            f"   • Num classes: {num_classes}"
        )
        
        return self.safe_execute(
            self._create_model_internal,
            "Gagal membuat model",
            backbone_type=backbone_type,
            pretrained=pretrained,
            num_classes=num_classes,
            detection_layers=detection_layers,
            **kwargs
        )
    
    def _create_model_internal(
        self,
        backbone_type: str,
        pretrained: bool,
        num_classes: int,
        detection_layers: List[str],
        **kwargs
    ) -> torch.nn.Module:
        """
        Implementasi internal untuk membuat model.
        
        Args:
            backbone_type: Tipe backbone
            pretrained: Gunakan pretrained weights
            num_classes: Jumlah kelas
            detection_layers: Layer deteksi
            
        Returns:
            Model yang dikonfigurasi
        """
        # Buat backbone
        backbone = self.backbone_factory.create_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained
        )
        
        # Impor model untuk menghindari circular import
        from smartcash.models.yolov5_model import YOLOv5Model
        
        # Buat model lengkap
        model = YOLOv5Model(
            backbone=backbone,
            num_classes=num_classes,
            backbone_type=backbone_type,
            detection_layers=detection_layers,
            logger=self.logger
        )
        
        self.logger.success(f"✅ Model berhasil dibuat dengan backbone {backbone_type}")
        return model
            
    def load_model(
        self, 
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint model
            device: Device untuk menempatkan model
            
        Returns:
            Tuple (Model yang dimuat, Metadata checkpoint)
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        return self.safe_execute(
            self._load_model_internal,
            f"Gagal memuat model dari {checkpoint_path}",
            checkpoint_path=checkpoint_path,
            device=device
        )
    
    def _load_model_internal(
        self,
        checkpoint_path: str,
        device: torch.device
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        Implementasi internal untuk memuat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            device: Device untuk model
            
        Returns:
            Tuple (Model, Checkpoint metadata)
        """
        # Import lazy untuk menghindari circular import
        from smartcash.handlers.checkpoint import CheckpointManager
        
        # Buat checkpoint manager
        checkpoint_manager = CheckpointManager(logger=self.logger)
        
        # Muat checkpoint
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        checkpoint_config = checkpoint.get('config', {})
        
        # Dapatkan informasi backbone dari checkpoint
        backbone = checkpoint_config.get('model', {}).get('backbone', 
                self.config.get('model', {}).get('backbone', 'efficientnet'))
        
        # Buat model baru dengan konfigurasi yang sama dengan checkpoint
        model = self.create_model(backbone_type=backbone)
        
        # Muat state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Pindahkan model ke device
        model = model.to(device)
        
        # Log informasi
        self.logger.success(
            f"✅ Model berhasil dimuat dari checkpoint:\n"
            f"   • Path: {checkpoint_path}\n"
            f"   • Epoch: {checkpoint.get('epoch', 'unknown')}\n"
            f"   • Backbone: {backbone}"
        )
        
        return model, checkpoint
    
    def freeze_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Bekukan layer backbone untuk fine-tuning.
        
        Args:
            model: Model dengan backbone yang akan dibekukan
            
        Returns:
            Model dengan backbone yang dibekukan
        """
        # Import YOLOv5Model untuk cek instance
        from smartcash.models.yolov5_model import YOLOv5Model
        
        if isinstance(model, YOLOv5Model) and hasattr(model, 'backbone'):
            # Bekukan backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            self.logger.info("🧊 Backbone telah dibekukan untuk fine-tuning")
            return model
        else:
            self.logger.warning("⚠️ Tipe model tidak dikenal, tidak dapat membekukan backbone")
            return model
    
    def unfreeze_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Lepaskan pembekuan layer backbone.
        
        Args:
            model: Model dengan backbone yang akan dilepas pembekuannya
            
        Returns:
            Model dengan backbone yang dilepas pembekuannya
        """
        # Import YOLOv5Model untuk cek instance
        from smartcash.models.yolov5_model import YOLOv5Model
        
        if isinstance(model, YOLOv5Model) and hasattr(model, 'backbone'):
            # Unfreeze backbone
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            self.logger.info("🔥 Backbone telah dilepas pembekuannya")
            return model
        else:
            self.logger.warning("⚠️ Tipe model tidak dikenal, tidak dapat melepas pembekuan backbone")
            return model