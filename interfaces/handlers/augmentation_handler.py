# File: src/interfaces/handlers/augmentation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk operasi augmentasi data mata uang

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from tqdm import tqdm

from interfaces.handlers.base_handler import BaseHandler

class AugmentationConfig:
    """Konfigurasi untuk augmentasi data"""
    def __init__(self):
        # Lighting variations
        self.brightness_range = (-0.3, 0.3)
        self.contrast_range = (-0.3, 0.3)
        self.gamma_range = (0.7, 1.3)
        
        # Geometric transformations
        self.rotation_range = (-30, 30)
        self.scale_range = (0.8, 1.2)
        self.shear_range = (-10, 10)
        
        # Currency specific
        self.blur_limit = 7
        self.noise_var = (10, 50)
        self.jpeg_quality = (60, 100)

class CurrencyAugmenter:
    """Augmenter khusus untuk mata uang"""
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        # Lighting augmentations
        self.lighting_aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_range,
                contrast_limit=self.config.contrast_range,
                p=0.8
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),  # Change to integer values between 50 and 200
                p=0.5
            ),
            A.CLAHE(p=0.3),
            A.RandomShadow(p=0.3)
        ], bbox_params=A.BboxParams(format='yolo'))
        
        # Geometric augmentations
        self.geometric_aug = A.Compose([
            A.SafeRotate(
                limit=self.config.rotation_range,
                p=0.8
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5
            ),
            A.RandomScale(
                scale_limit=self.config.scale_range,
                p=0.5
            )
        ], bbox_params=A.BboxParams(format='yolo'))
        
        # Currency condition augmentations
        self.condition_aug = A.Compose([
            A.GaussianBlur(
                blur_limit=self.config.blur_limit,
                p=0.5
            ),
            A.GaussNoise(
                var_limit=self.config.noise_var,
                p=0.5
            ),
            A.ImageCompression(
                quality_lower=self.config.jpeg_quality[0],
                quality_upper=self.config.jpeg_quality[1],
                p=0.5
            )
        ], bbox_params=A.BboxParams(format='yolo'))

class DataAugmentationHandler(BaseHandler):
    """Handler untuk operasi augmentasi dataset"""
    def __init__(self, config):
        super().__init__(config)
        self.aug_config = AugmentationConfig()
        self.augmenter = CurrencyAugmenter(self.aug_config)
        
    def augment_dataset(self, 
                       factor: int = 2, 
                       modes: Optional[List[str]] = None) -> Dict:
        """
        Augmentasi dataset dengan faktor dan mode tertentu
        
        Args:
            factor: Faktor penggandaan data
            modes: List mode augmentasi ('lighting', 'geometric', 'condition')
        """
        self.logger.info(f"🎨 Memulai augmentasi dataset (faktor: {factor}x)")
        
        stats = {
            'processed': 0,
            'augmented': 0,
            'errors': 0
        }
        
        try:
            # Validasi modes
            valid_modes = ['lighting', 'geometric', 'condition']
            if modes:
                modes = [m for m in modes if m in valid_modes]
            else:
                modes = valid_modes
                
            # Process training data
            train_img_dir = self.rupiah_dir / 'train' / 'images'
            train_label_dir = self.rupiah_dir / 'train' / 'labels'
            
            if not train_img_dir.exists() or not train_label_dir.exists():
                raise Exception("Direktori training tidak ditemukan")
                
            # Get original files
            img_files = [f for f in train_img_dir.glob('*.jpg') 
                        if '_aug' not in f.stem]
                        
            # Process each image
            with tqdm(total=len(img_files) * factor, 
                     desc="Mengaugmentasi gambar") as pbar:
                for img_path in img_files:
                    try:
                        stats['processed'] += 1
                        
                        # Load image and label
                        img = cv2.imread(str(img_path))
                        label_path = train_label_dir / f"{img_path.stem}.txt"
                        
                        if img is None or not label_path.exists():
                            continue
                            
                        labels = self._load_labels(label_path)
                        
                        # Generate augmentations
                        for i in range(factor):
                            aug_img, aug_labels = self._apply_augmentation(
                                img, labels, modes
                            )
                            
                            # Save augmented data
                            if self._save_augmented_data(
                                aug_img, aug_labels,
                                img_path.stem, i,
                                train_img_dir, train_label_dir
                            ):
                                stats['augmented'] += 1
                                
                            pbar.update(1)
                            
                    except Exception as e:
                        self.logger.error(f"Error pada {img_path.name}: {str(e)}")
                        stats['errors'] += 1
                        
            # Update operation stats
            self.update_stats('augmentation', stats)
            self.log_operation("Augmentasi dataset", "success")
            
        except Exception as e:
            self.log_operation("Augmentasi dataset", "failed", str(e))
            
        return stats
        
    def _load_labels(self, path: Path) -> np.ndarray:
        """Load label file"""
        labels = []
        with open(path) as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                labels.append(values)
        return np.array(labels)
        
    def _apply_augmentation(self, 
                          img: np.ndarray, 
                          labels: np.ndarray,
                          modes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation sequence"""
        data = {'image': img, 'bboxes': labels[:, 1:], 'class_labels': labels[:, 0]}
        
        # Apply selected augmentations
        if 'lighting' in modes:
            transformed = self.augmenter.lighting_aug(**data)
            data = {
                'image': transformed['image'],
                'bboxes': transformed['bboxes'],
                'class_labels': transformed['class_labels']
            }
            
        if 'geometric' in modes:
            transformed = self.augmenter.geometric_aug(**data)
            data = {
                'image': transformed['image'],
                'bboxes': transformed['bboxes'],
                'class_labels': transformed['class_labels']
            }
            
        if 'condition' in modes:
            transformed = self.augmenter.condition_aug(**data)
            
        # Combine labels
        aug_labels = np.column_stack([
            transformed['class_labels'],
            transformed['bboxes']
        ])
        
        return transformed['image'], aug_labels
        
    def _save_augmented_data(self,
                           img: np.ndarray,
                           labels: np.ndarray,
                           base_name: str,
                           idx: int,
                           img_dir: Path,
                           label_dir: Path) -> bool:
        """Save augmented image and labels"""
        try:
            # Save image
            aug_img_path = img_dir / f"{base_name}_aug{idx}.jpg"
            cv2.imwrite(str(aug_img_path), img)
            
            # Save labels
            aug_label_path = label_dir / f"{base_name}_aug{idx}.txt"
            with open(aug_label_path, 'w') as f:
                for label in labels:
                    f.write(' '.join(map(str, label)) + '\n')
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Gagal menyimpan augmentasi: {str(e)}")
            return False
            
    def get_augmentation_descriptions(self) -> Dict:
        """Get descriptions of available augmentation types"""
        return {
            'lighting': {
                'name': 'Augmentasi Pencahayaan',
                'description': 'Simulasi variasi pencahayaan',
                'features': [
                    'Variasi kecerahan dan kontras',
                    'Koreksi gamma',
                    'Penyesuaian histogram',
                    'Simulasi bayangan'
                ],
                'use_case': 'Kondisi pencahayaan berbeda'
            },
            'geometric': {
                'name': 'Augmentasi Geometrik',
                'description': 'Simulasi variasi posisi dan orientasi',
                'features': [
                    'Rotasi aman dengan padding',
                    'Scaling adaptif',
                    'Transformasi perspektif',
                    'Pergeseran posisi'
                ],
                'use_case': 'Variasi sudut pandang kamera'
            },
            'condition': {
                'name': 'Augmentasi Kondisi',
                'description': 'Simulasi kondisi uang dan kamera',
                'features': [
                    'Blur gaussian untuk uang lusuh',
                    'Noise untuk kondisi low-light',
                    'Kompresi JPEG untuk kualitas kamera',
                    'Tekstur untuk lipatan'
                ],
                'use_case': 'Variasi kondisi uang dan pengambilan gambar'
            }
        }
        
    def get_recommended_factor(self) -> int:
        """Get recommended augmentation factor based on dataset size"""
        train_img_dir = self.rupiah_dir / 'train' / 'images'
        if not train_img_dir.exists():
            return 2
            
        # Count original images
        original_count = len([f for f in train_img_dir.glob('*.jpg')
                            if '_aug' not in f.stem])
                            
        if original_count > 1000:
            return 2  # Large dataset
        elif original_count > 500:
            return 4  # Medium dataset
        else:
            return 6  # Small dataset