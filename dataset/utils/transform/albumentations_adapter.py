"""
File: smartcash/dataset/utils/transform/albumentations_adapter.py
Deskripsi: Adapter untuk library Albumentations dengan dukungan transformasi dataset SmartCash
"""

import cv2
import numpy as np
import albumentations as A
from typing import Dict, Tuple, Optional, Any, List, Union

from smartcash.common.logger import get_logger


class AlbumentationsAdapter:
    """Adapter untuk mengintegrasikan Albumentations dengan dataset SmartCash."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi AlbumentationsAdapter.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("albumentations_adapter")
    
    def get_basic_transforms(
        self, 
        img_size: Tuple[int, int] = (640, 640),
        normalize: bool = True
    ) -> A.Compose:
        """
        Dapatkan transformasi dasar (resize & normalize).
        
        Args:
            img_size: Ukuran target gambar
            normalize: Apakah menggunakan normalisasi
            
        Returns:
            Objek Compose dari Albumentations
        """
        transforms = [
            A.Resize(height=img_size[1], width=img_size[0], p=1.0)
        ]
        
        if normalize:
            transforms.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
            )
            
        return A.Compose(transforms)
    
    def get_geometric_augmentations(
        self,
        img_size: Tuple[int, int] = (640, 640),
        p: float = 0.5,
        scale: Tuple[float, float] = (0.8, 1.0),
        fliplr_p: float = 0.5,
        shift_limit: float = 0.1,
        scale_limit: float = 0.1,
        rotate_limit: int = 15,
        with_bbox: bool = True
    ) -> A.Compose:
        """
        Dapatkan augmentasi geometrik (perubahan posisi, rotasi, skala, flip).
        
        Args:
            img_size: Ukuran target gambar
            p: Probabilitas menerapkan augmentasi
            scale: Range scale untuk RandomResizedCrop
            fliplr_p: Probabilitas horizontal flip
            shift_limit: Limit pergeseran pada ShiftScaleRotate
            scale_limit: Limit skala pada ShiftScaleRotate
            rotate_limit: Limit rotasi pada ShiftScaleRotate (derajat)
            with_bbox: Apakah termasuk parameter untuk bbox
            
        Returns:
            Objek Compose dari Albumentations
        """
        transforms = [
            A.RandomResizedCrop(height=img_size[1], width=img_size[0], scale=scale, p=1.0),
            A.HorizontalFlip(p=fliplr_p),
            A.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                p=p,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]
        
        if with_bbox:
            return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        else:
            return A.Compose(transforms)
    
    def get_color_augmentations(
        self,
        p: float = 0.5,
        brightness_limit: float = 0.3,
        contrast_limit: float = 0.3,
        hue_shift_limit: int = 5,
        sat_shift_limit: int = 30,
        val_shift_limit: int = 20,
        with_bbox: bool = True
    ) -> A.Compose:
        """
        Dapatkan augmentasi warna (brightness, contrast, hue, saturation, value).
        
        Args:
            p: Probabilitas menerapkan augmentasi
            brightness_limit: Limit perubahan brightness
            contrast_limit: Limit perubahan contrast
            hue_shift_limit: Limit perubahan hue (derajat)
            sat_shift_limit: Limit perubahan saturation
            val_shift_limit: Limit perubahan value
            with_bbox: Apakah termasuk parameter untuk bbox
            
        Returns:
            Objek Compose dari Albumentations
        """
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=p),
            A.HueSaturationValue(
                hue_shift_limit=hue_shift_limit,
                sat_shift_limit=sat_shift_limit,
                val_shift_limit=val_shift_limit,
                p=p
            )
        ]
        
        if with_bbox:
            return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose(transforms)
    
    def get_noise_augmentations(
        self,
        p: float = 0.3,
        with_bbox: bool = True
    ) -> A.Compose:
        """
        Dapatkan augmentasi noise (blur, noise).
        
        Args:
            p: Probabilitas menerapkan augmentasi
            with_bbox: Apakah termasuk parameter untuk bbox
            
        Returns:
            Objek Compose dari Albumentations
        """
        transforms = [
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0)
            ], p=p),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
            ], p=p)
        ]
        
        if with_bbox:
            return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose(transforms)
    
    def create_augmentation_pipeline(
        self,
        img_size: Tuple[int, int] = (640, 640),
        augmentation_types: List[str] = None,
        with_bbox: bool = True
    ) -> A.Compose:
        """
        Buat pipeline augmentasi berdasarkan jenis yang dipilih.
        
        Args:
            img_size: Ukuran target gambar
            augmentation_types: Jenis augmentasi ('geometric', 'color', 'noise')
            with_bbox: Apakah termasuk parameter untuk bbox
            
        Returns:
            Objek Compose dari Albumentations
        """
        if augmentation_types is None:
            augmentation_types = ['geometric', 'color']
            
        transforms = []
        
        # Selalu tambahkan RandomResizedCrop
        transforms.append(A.RandomResizedCrop(height=img_size[1], width=img_size[0], scale=(0.8, 1.0), p=1.0))
        
        # Tambahkan augmentasi berdasarkan tipe
        if 'geometric' in augmentation_types:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT
                )
            ])
            
        if 'color' in augmentation_types:
            transforms.extend([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                )
            ])
            
        if 'noise' in augmentation_types:
            transforms.append(
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(var_limit=(10, 50), p=1.0)
                ], p=0.3)
            )
            
        # Selalu tambahkan normalisasi di akhir
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0))
        
        if with_bbox:
            return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        else:
            return A.Compose(transforms)
            
    def apply_transforms(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        class_labels: Optional[List[int]] = None,
        transforms: Optional[A.Compose] = None,
        img_size: Tuple[int, int] = (640, 640)
    ) -> Dict[str, Any]:
        """
        Terapkan transforms ke gambar dan bbox.
        
        Args:
            image: Array gambar (H, W, C)
            bboxes: List bounding box dalam format YOLO (opsional)
            class_labels: List class ID untuk bboxes (opsional)
            transforms: Transforms yang akan diterapkan (opsional)
            img_size: Ukuran target gambar (jika transforms tidak disediakan)
            
        Returns:
            Dictionary hasil transformasi
        """
        if transforms is None:
            transforms = self.get_basic_transforms(img_size)
            
        if bboxes and class_labels and len(bboxes) > 0:
            # Terapkan transformasi dengan bbox
            try:
                transformed = transforms(image=image, bboxes=bboxes, class_labels=class_labels)
                return transformed
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat menerapkan transforms dengan bbox: {str(e)}")
                # Fallback ke transformasi tanpa bbox
                transformed = transforms(image=image)
                transformed['bboxes'] = bboxes
                transformed['class_labels'] = class_labels
                return transformed
        else:
            # Terapkan transformasi tanpa bbox
            transformed = transforms(image=image)
            transformed['bboxes'] = bboxes or []
            transformed['class_labels'] = class_labels or []
            return transformed