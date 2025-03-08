# File: smartcash/handlers/model/model_factory.py
# Author: Alfrida Sabar
# Deskripsi: Kelas factory untuk membuat model dengan backbone yang berbeda

import torch
from typing import Dict, Optional, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.baseline import BaselineModel

class ModelFactory:
    """
    Factory class untuk membuat model dengan backbone yang berbeda.
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
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        self.num_classes = config.get('model', {}).get('num_classes', 7)
    
    def create_model(self, backbone_type: Optional[str] = None) -> Union[YOLOv5Model, BaselineModel]:
        """
        Membuat model dengan backbone dan konfigurasi yang ditentukan.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            
        Returns:
            Model yang siap digunakan
        """
        # Prioritaskan backbone yang diberikan, atau gunakan dari config
        backbone_type = backbone_type or self.config.get('model', {}).get('backbone', 'cspdarknet')
        
        # Parameter lain dari config
        pretrained = self.config.get('model', {}).get('pretrained', True)
        detection_layers = self.config.get('layers', ['banknote'])
        
        # Log informasi model
        self.logger.info(
            f"🔄 Membuat model dengan backbone: {backbone_type}\n"
            f"   • Pretrained: {pretrained}\n"
            f"   • Detection layers: {detection_layers}\n"
            f"   • Num classes: {self.num_classes}"
        )
        
        # Buat model sesuai tipe backbone
        try:
            if backbone_type in ['efficientnet', 'cspdarknet']:
                model = YOLOv5Model(
                    num_classes=self.num_classes,
                    backbone_type=backbone_type,
                    pretrained=pretrained,
                    detection_layers=detection_layers,
                    logger=self.logger
                )
            else:
                # Fallback untuk backbone lain
                model = BaselineModel(
                    num_classes=self.num_classes,
                    backbone=backbone_type,
                    pretrained=pretrained
                )
                
            self.logger.success(f"✅ Model berhasil dibuat dengan backbone {backbone_type}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat model: {str(e)}")
            raise e
            
    def freeze_backbone(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Bekukan layer backbone untuk fine-tuning.
        
        Args:
            model: Model dengan backbone yang akan dibekukan
            
        Returns:
            Model dengan backbone yang dibekukan
        """
        if isinstance(model, YOLOv5Model):
            # Bekukan backbone YOLOv5
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            self.logger.info("🧊 Backbone telah dibekukan untuk fine-tuning")
            return model
        elif isinstance(model, BaselineModel):
            # Bekukan backbone BaselineModel
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
        if isinstance(model, YOLOv5Model):
            # Unfreeze backbone YOLOv5
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            self.logger.info("🔥 Backbone telah dilepas pembekuannya")
            return model
        elif isinstance(model, BaselineModel):
            # Unfreeze backbone BaselineModel
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            self.logger.info("🔥 Backbone telah dilepas pembekuannya")
            return model
        else:
            self.logger.warning("⚠️ Tipe model tidak dikenal, tidak dapat melepas pembekuan backbone")
            return model