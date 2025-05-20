"""
File: smartcash/dataset/utils/transform/image_transform.py
Deskripsi: Komponen untuk transformasi dan augmentasi gambar
"""

import cv2
import numpy as np
import albumentations as A
from typing import Dict, Tuple, Optional, Any, List, Union

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.transform.bbox_transform import BBoxTransformer


class ImageTransformer:
    """Transformasi dan augmentasi gambar untuk dataset SmartCash."""
    
    def __init__(self, config: Dict, img_size: Tuple[int, int] = (640, 640), logger=None):
        """
        Inisialisasi ImageTransformer.
        
        Args:
            config: Konfigurasi aplikasi
            img_size: Ukuran target gambar
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.img_size = img_size
        self.logger = logger or get_logger()
        
        # Setup transformasi dasar
        self._setup_transformations()
        self.logger.info(f"🎨 ImageTransformer diinisialisasi dengan target size: {img_size}")
    
    def _setup_transformations(self) -> None:
        """Setup berbagai pipeline transformasi."""
        # Ambil konfigurasi dari config
        aug_config = self.config.get('augmentation', {})
        training_config = self.config.get('training', {})
        
        # Parameter dasar untuk augmentasi
        position_params = aug_config.get('position', {})
        lighting_params = aug_config.get('lighting', {})
        
        # Transformasi untuk training (dengan augmentasi)
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=self.img_size[1], width=self.img_size[0], scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=position_params.get('fliplr', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=position_params.get('translate', 0.1),
                scale_limit=position_params.get('scale', 0.1),
                rotate_limit=position_params.get('degrees', 15),
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(
                brightness_limit=lighting_params.get('brightness', 0.3),
                contrast_limit=lighting_params.get('contrast', 0.3),
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=lighting_params.get('hsv_h', 0.015) * 360,
                sat_shift_limit=lighting_params.get('hsv_s', 0.7) * 255,
                val_shift_limit=lighting_params.get('hsv_v', 0.4) * 255,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Transformasi untuk validasi dan testing (tanpa augmentasi)
        self.val_transform = A.Compose([
            A.Resize(height=self.img_size[1], width=self.img_size[0], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Transformasi untuk inferensi (tanpa bbox)
        self.inference_transform = A.Compose([
            A.Resize(height=self.img_size[1], width=self.img_size[0], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ])
    
    def get_transform(self, mode: str = 'train') -> A.Compose:
        """
        Dapatkan transformasi sesuai mode.
        
        Args:
            mode: Mode transformasi ('train', 'valid'/'val', 'test'/'inference')
            
        Returns:
            Objek Compose dari Albumentations
        """
        mode = mode.lower()
        if mode == 'train':
            return self.train_transform
        elif mode in ('val', 'valid', 'validation'):
            return self.val_transform
        elif mode in ('test', 'inference'):
            return self.inference_transform
        else:
            self.logger.warning(f"⚠️ Mode transformasi '{mode}' tidak dikenal, menggunakan val_transform")
            return self.val_transform
    
    def create_custom_transform(self, **kwargs) -> A.Compose:
        """
        Buat transformasi kustom dengan parameter yang disesuaikan.
        
        Args:
            **kwargs: Parameter kustom untuk transformasi
            
        Returns:
            Objek Compose dari Albumentations
        """
        # Ambil parameter augmentasi
        aug_config = self.config.get('augmentation', {})
        position_params = aug_config.get('position', {})
        lighting_params = aug_config.get('lighting', {})
        
        # Override parameter dengan kwargs jika disediakan
        fliplr = kwargs.get('fliplr', position_params.get('fliplr', 0.5))
        translate = kwargs.get('translate', position_params.get('translate', 0.1))
        scale = kwargs.get('scale', position_params.get('scale', 0.1))
        degrees = kwargs.get('degrees', position_params.get('degrees', 15))
        brightness = kwargs.get('brightness', lighting_params.get('brightness', 0.3))
        contrast = kwargs.get('contrast', lighting_params.get('contrast', 0.3))
        hsv_h = kwargs.get('hsv_h', lighting_params.get('hsv_h', 0.015))
        hsv_s = kwargs.get('hsv_s', lighting_params.get('hsv_s', 0.7))
        hsv_v = kwargs.get('hsv_v', lighting_params.get('hsv_v', 0.4))
        
        # Log parameter
        self.logger.info(
            f"🎨 Membuat transform kustom:\n"
            f"   • Position: fliplr={fliplr}, translate={translate}, scale={scale}, degrees={degrees}\n"
            f"   • Lighting: brightness={brightness}, contrast={contrast}, hsv=({hsv_h}, {hsv_s}, {hsv_v})"
        )
        
        # Buat transform
        return A.Compose([
            A.RandomResizedCrop(height=self.img_size[1], width=self.img_size[0], scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=fliplr),
            A.ShiftScaleRotate(
                shift_limit=translate, 
                scale_limit=scale, 
                rotate_limit=degrees,
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=hsv_h * 360,
                sat_shift_limit=hsv_s * 255,
                val_shift_limit=hsv_v * 255,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    def process_image(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        class_labels: Optional[List[int]] = None,
        mode: str = 'train'
    ) -> Dict[str, Any]:
        """
        Proses gambar dengan transform yang sesuai.
        
        Args:
            image: Array gambar (H, W, C)
            bboxes: Lista bounding box dalam format YOLO (opsional)
            class_labels: List class ID untuk bboxes (opsional)
            mode: Mode transformasi
            
        Returns:
            Dictionary hasil transformasi
        """
        transform = self.get_transform(mode)
        
        if bboxes and class_labels and len(bboxes) > 0:
            # Terapkan transformasi dengan bbox
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed
        else:
            # Terapkan transformasi tanpa bbox
            if mode == 'train':
                # Gunakan versi tanpa bbox_params untuk mode train
                transform = A.Compose([t for t in transform if not isinstance(t, A.Normalize)] + 
                                     [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)])
            
            transformed = transform(image=image)
            transformed['bboxes'] = bboxes or []
            transformed['class_labels'] = class_labels or []
            return transformed
    
    def get_normalization_params(self) -> Dict[str, List[float]]:
        """
        Dapatkan parameter normalisasi untuk inferensi.
        
        Returns:
            Dictionary berisi parameter normalisasi
        """
        return {
            'mean': [0.485, 0.456, 0.406], 
            'std': [0.229, 0.224, 0.225]
        }
    
    def resize_image(self, image: np.ndarray, keep_ratio: bool = True) -> np.ndarray:
        """
        Resize gambar ke target size.
        
        Args:
            image: Array gambar (H, W, C)
            keep_ratio: Pertahankan aspek rasio asli
            
        Returns:
            Gambar yang telah di-resize
        """
        if keep_ratio:
            # Maintain aspect ratio
            h, w = image.shape[:2]
            target_h, target_w = self.img_size
            
            # Calculate scale
            scale = min(target_h / h, target_w / w)
            
            # Calculate new dimensions
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create canvas of target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center image on canvas
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return canvas
        else:
            # Simple resize
            return cv2.resize(image, (self.img_size[0], self.img_size[1]))