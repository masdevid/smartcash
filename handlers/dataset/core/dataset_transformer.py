# File: smartcash/handlers/dataset/core/dataset_transformer.py
# Author: Alfrida Sabar
# Deskripsi: Transformasi untuk dataset SmartCash

import albumentations as A
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger


class DatasetTransformer:
    """
    Komponen untuk transformasi dataset SmartCash.
    Menyediakan transformasi yang konsisten untuk training, validasi, dan testing.
    """
    
    def __init__(
        self,
        config: Dict,
        img_size: Tuple[int, int] = (640, 640),
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetTransformer.
        
        Args:
            config: Konfigurasi augmentasi/training
            img_size: Ukuran target gambar
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.img_size = img_size
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup pipeline transformasi
        self._setup_transformations()
        
        self.logger.info(f"🔄 DatasetTransformer diinisialisasi dengan target size: {img_size}")
    
    def _setup_transformations(self) -> None:
        """Setup berbagai pipeline transformasi."""
        train_config = self.config.get('training', {})
        
        # Augmentasi untuk training
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=self.img_size[1],
                width=self.img_size[0],
                scale=(0.8, 1.0),
                p=1.0
            ),
            A.HorizontalFlip(p=train_config.get('fliplr', 0.5)),
            A.VerticalFlip(p=train_config.get('flipud', 0.0)),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=train_config.get('hsv_h', 0.015),
                sat_shift_limit=train_config.get('hsv_s', 0.7),
                val_shift_limit=train_config.get('hsv_v', 0.4),
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=train_config.get('translate', 0.1),
                scale_limit=train_config.get('scale', 0.5),
                rotate_limit=train_config.get('degrees', 0.0),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Transformasi minimum untuk validasi/testing
        self.val_transform = A.Compose([
            A.Resize(height=self.img_size[1], width=self.img_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        # Transformasi khusus untuk fase testing yang mungkin memerlukan post-processing
        self.test_transform = A.Compose([
            A.Resize(height=self.img_size[1], width=self.img_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        # Transformasi untuk inferensi tanpa label
        self.inference_transform = A.Compose([
            A.Resize(height=self.img_size[1], width=self.img_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ])
        
        self.logger.info(f"✅ Pipeline transformasi berhasil disiapkan")
    
    def get_transform(self, mode: str = 'train') -> A.Compose:
        """
        Mendapatkan transformasi sesuai mode.
        
        Args:
            mode: Mode transformasi ('train', 'val', 'test', 'inference')
            
        Returns:
            Pipeline transformasi albumentations
        """
        mode = mode.lower()
        
        if mode == 'train':
            return self.train_transform
        elif mode in ('val', 'valid', 'validation'):
            return self.val_transform
        elif mode == 'test':
            return self.test_transform
        elif mode == 'inference':
            return self.inference_transform
        else:
            self.logger.warning(f"⚠️ Mode transformasi '{mode}' tidak dikenal, menggunakan val_transform")
            return self.val_transform
    
    def create_custom_train_transform(self, **kwargs) -> A.Compose:
        """
        Membuat transformasi training kustom dengan parameter yang disesuaikan.
        
        Args:
            **kwargs: Parameter kustom yang akan digunakan untuk mengganti default
            
        Returns:
            Pipeline transformasi albumentations kustom
        """
        train_config = self.config.get('training', {})
        
        # Ambil parameter dari kwargs atau config atau nilai default
        fliplr = kwargs.get('fliplr', train_config.get('fliplr', 0.5))
        flipud = kwargs.get('flipud', train_config.get('flipud', 0.0))
        translate = kwargs.get('translate', train_config.get('translate', 0.1))
        scale = kwargs.get('scale', train_config.get('scale', 0.5))
        degrees = kwargs.get('degrees', train_config.get('degrees', 45.0))
        hsv_h = kwargs.get('hsv_h', train_config.get('hsv_h', 0.015))
        hsv_s = kwargs.get('hsv_s', train_config.get('hsv_s', 0.7))
        hsv_v = kwargs.get('hsv_v', train_config.get('hsv_v', 0.4))
        
        self.logger.info(
            f"🎨 Membuat transformasi kustom dengan:\n"
            f"   • fliplr={fliplr}, flipud={flipud}\n"
            f"   • translate={translate}, scale={scale}, degrees={degrees}\n"
            f"   • hsv_h={hsv_h}, hsv_s={hsv_s}, hsv_v={hsv_v}"
        )
        
        # Buat pipeline kustom
        return A.Compose([
            A.RandomResizedCrop(
                height=self.img_size[1],
                width=self.img_size[0],
                scale=(0.8, 1.0),
                p=1.0
            ),
            A.HorizontalFlip(p=fliplr),
            A.VerticalFlip(p=flipud),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=hsv_h,
                sat_shift_limit=hsv_s,
                val_shift_limit=hsv_v,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=translate,
                scale_limit=scale,
                rotate_limit=degrees,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
    def get_preprocessing_params(self) -> Dict:
        """
        Mendapatkan parameter preprocessing untuk normalisasi.
        
        Returns:
            Dict yang berisi mean dan std
        """
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }