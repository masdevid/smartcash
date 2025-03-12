"""
File: smartcash/utils/augmentation/augmentation_processor.py
Author: Alfrida Sabar (fixed)
Deskripsi: Processor augmentasi dengan perbaikan untuk warning bbox processor.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import albumentations as A

from smartcash.utils.logger import get_logger
from smartcash.utils.coordinate_utils import CoordinateUtils

class AugmentationProcessor:
    """Processor untuk transformasi augmentasi data dengan dukungan yolo bbox."""
    
    def __init__(self, config, pipeline=None, output_dir=None, logger=None):
        """Inisialisasi processor augmentasi."""
        self.config = config
        self.pipeline = pipeline
        self.output_dir = Path(output_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("aug_processor")
        self.coord_utils = CoordinateUtils()
        
        # Default augmentation settings
        self.aug_settings = config.get('augmentation', {})
        
    def _get_transform(self, transform_type: str) -> A.Compose:
        """
        Dapatkan transform berdasarkan jenis augmentasi.
        
        Args:
            transform_type: Jenis augmentasi ("position", "lighting", "combined", "extreme_rotation")
            
        Returns:
            Compose object dari Albumentations
        """
        # Default settings
        pos_settings = self.aug_settings.get('position', {})
        light_settings = self.aug_settings.get('lighting', {})
        
        # Transform dasar yang selalu digunakan
        base_transform = [
            A.Resize(height=640, width=640, p=1.0)
        ]
        
        # Position augmentations
        if transform_type in ["position", "combined"]:
            position_transforms = [
                A.HorizontalFlip(p=pos_settings.get('fliplr', 0.5)),
                A.ShiftScaleRotate(
                    shift_limit=pos_settings.get('translate', 0.1),
                    scale_limit=pos_settings.get('scale', 0.1),
                    rotate_limit=pos_settings.get('degrees', 15),
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT
                )
            ]
        else:
            position_transforms = []
            
        # Lighting augmentations
        if transform_type in ["lighting", "combined"]:
            lighting_transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=light_settings.get('brightness', 0.3),
                    contrast_limit=light_settings.get('contrast', 0.3),
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=light_settings.get('hsv_h', 0.015) * 360,
                    sat_shift_limit=light_settings.get('hsv_s', 0.7) * 255,
                    val_shift_limit=light_settings.get('hsv_v', 0.4) * 255,
                    p=0.5
                )
            ]
        else:
            lighting_transforms = []
            
        # Extreme rotation (special case)
        if transform_type == "extreme_rotation":
            extreme_settings = self.aug_settings.get('extreme', {})
            extreme_transforms = [
                A.SafeRotate(
                    limit=extreme_settings.get('rotation_max', 90),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.8
                )
            ]
        else:
            extreme_transforms = []
        
        # Gabungkan transform berdasarkan tipe
        all_transforms = base_transform + position_transforms + lighting_transforms + extreme_transforms
        
        # Pastikan ada transform untuk bboxes dengan menambahkan bbox_params
        # FIX: Menambahkan bbox_params untuk menghindari warning "Got processor for bboxes, but no transform to process it"
        return A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                min_area=1e-5,
                min_visibility=0.1,
                label_fields=['class_labels']
            )
        )
    
    def augment_image(
        self, 
        image_path: Union[str, Path], 
        label_path: Optional[Union[str, Path]] = None, 
        aug_type: str = 'combined',
        output_prefix: str = 'aug',
        variations: int = 2
    ) -> Tuple[List[np.ndarray], List[Dict], List[Path]]:
        """
        Augmentasi gambar dan label YOLO dengan transformasi tertentu.
        
        Args:
            image_path: Path gambar input
            label_path: Path label YOLO (opsional)
            aug_type: Jenis augmentasi ("position", "lighting", "combined", "extreme_rotation")
            output_prefix: Prefix untuk nama file output
            variations: Jumlah variasi yang akan dibuat
            
        Returns:
            Tuple berisi (augmented_images, augmented_labels, output_paths)
        """
        # Validasi input
        image_path = Path(image_path)
        if not image_path.exists():
            self.logger.warning(f"⚠️ Gambar tidak ditemukan: {image_path}")
            return [], [], []
        
        # Baca gambar
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"⚠️ Gagal membaca gambar: {image_path}")
                return [], [], []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.warning(f"⚠️ Error membaca gambar {image_path}: {str(e)}")
            return [], [], []
            
        # Dapatkan bboxes dan class_ids dari file label jika ada
        bboxes = []
        class_ids = []
        
        if label_path and Path(label_path).exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            
                            # Validasi koordinat
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                bboxes.append([x, y, w, h])
                                class_ids.append(class_id)
            except Exception as e:
                self.logger.warning(f"⚠️ Error membaca label {label_path}: {str(e)}")
        
        # Dapatkan transform
        transform = self._get_transform(aug_type)
        
        # Buat hasil augmentasi
        augmented_images = []
        augmented_labels = []
        output_paths = []
        
        # Generate augmentasi sebanyak variations
        for i in range(variations):
            try:
                # Apply transformasi
                if bboxes:
                    # Dengan bboxes
                    result = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_ids
                    )
                    
                    aug_image = result['image']
                    aug_bboxes = result['bboxes']
                    aug_class_ids = result['class_labels']
                    
                    # Convert bbox ke format yolo
                    aug_labels = {}
                    for j, (box, cls_id) in enumerate(zip(aug_bboxes, aug_class_ids)):
                        if len(box) == 4:  # Pastikan bbox valid
                            if cls_id not in aug_labels:
                                aug_labels[cls_id] = []
                            aug_labels[cls_id].append(box)
                else:
                    # Tanpa bboxes
                    result = transform(image=image)
                    aug_image = result['image']
                    aug_labels = {}
                
                # Generate nama file output
                output_name = f"{output_prefix}_{aug_type}_{i}_{image_path.stem}{image_path.suffix}"
                output_path = Path(output_name)
                
                # Simpan hasilnya
                augmented_images.append(aug_image)
                augmented_labels.append(aug_labels)
                output_paths.append(output_path)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat augmentasi {aug_type} #{i}: {str(e)}")
        
        return augmented_images, augmented_labels, output_paths
    
    def save_augmented_data(
        self, 
        image: np.ndarray, 
        labels: Dict[int, List[List[float]]], 
        image_path: Path,
        labels_dir: Path
    ) -> bool:
        """
        Simpan gambar hasil augmentasi dan labelnya.
        
        Args:
            image: Gambar hasil augmentasi
            labels: Dictionary label hasil augmentasi (class_id -> [bbox])
            image_path: Path untuk menyimpan gambar
            labels_dir: Direktori untuk menyimpan label
            
        Returns:
            Boolean yang menunjukkan keberhasilan save
        """
        try:
            # Pastikan direktori ada
            image_path.parent.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Simpan gambar
            cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), cv2_image)
            
            # Simpan label
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                for class_id, bboxes in labels.items():
                    for bbox in bboxes:
                        if len(bbox) == 4:  # Ensure the bbox is valid
                            x, y, w, h = bbox
                            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat menyimpan hasil augmentasi: {str(e)}")
            return False