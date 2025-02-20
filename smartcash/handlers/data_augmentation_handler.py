# File: handlers/data_augmentation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk augmentasi data dengan fokus pada variasi pencahayaan 
# dan posisi pengambilan gambar uang kertas Rupiah

import os
from typing import Dict, List, Optional, Tuple
import albumentations as A
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger

class DataAugmentationHandler:
    """Handler untuk augmentasi data dengan fokus pada pencahayaan dan posisi"""
    
    def __init__(
        self,
        output_dir: str = "data/augmented",
        n_workers: int = 4,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        
        # Buat output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup augmentation pipelines
        self.position_aug = self._setup_position_augmentation()
        self.lighting_aug = self._setup_lighting_augmentation()
        
    def _setup_position_augmentation(self) -> A.Compose:
        """Setup augmentasi untuk variasi posisi"""
        return A.Compose([
            A.SafeRotate(limit=30, p=0.7),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
    def _setup_lighting_augmentation(self) -> A.Compose:
        """Setup augmentasi untuk variasi pencahayaan"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.OneOf([
                A.RandomShadow(p=0.5),
                A.RandomToneCurve(p=0.5)
            ], p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=80,
                quality_upper=100,
                p=0.2
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
    def augment_image(
        self,
        image_path: Path,
        label_path: Optional[Path] = None,
        augmentation_type: str = 'position'
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Augmentasi satu gambar
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (opsional)
            augmentation_type: Tipe augmentasi ('position' atau 'lighting')
        Returns:
            Tuple berisi list gambar dan label hasil augmentasi
        """
        # Baca gambar
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Baca label jika ada
        bboxes = []
        class_labels = []
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    cls, *bbox = map(float, line.strip().split())
                    bboxes.append(bbox)
                    class_labels.append(cls)
        
        # Pilih pipeline augmentasi
        augmentor = (self.position_aug if augmentation_type == 'position' 
                    else self.lighting_aug)
        
        # Generate 3 variasi
        augmented_images = []
        augmented_bboxes = []
        
        for _ in range(3):
            augmented = augmentor(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            augmented_images.append(augmented['image'])
            if bboxes:
                augmented_bboxes.append(
                    np.array([
                        [cls, *box] 
                        for cls, box in zip(
                            augmented['class_labels'],
                            augmented['bboxes']
                        )
                    ])
                )
                
        return augmented_images, augmented_bboxes
        
    def augment_dataset(
        self,
        input_dir: str,
        augmentation_type: str = 'position'
    ) -> Dict:
        """
        Augmentasi seluruh dataset
        Args:
            input_dir: Direktori dataset
            augmentation_type: Tipe augmentasi ('position' atau 'lighting')
        Returns:
            Dict statistik augmentasi
        """
        input_path = Path(input_dir)
        stats = {'original': 0, 'augmented': 0, 'failed': 0}
        
        self.logger.start(
            f"🎨 Memulai augmentasi dataset ({augmentation_type})\n"
            f"📁 Input: {input_dir}\n"
            f"📁 Output: {self.output_dir}"
        )
        
        try:
            # Collect semua file gambar
            image_files = list(input_path.glob('images/*.jpg'))
            stats['original'] = len(image_files)
            
            # Progress bar
            with tqdm(total=len(image_files), 
                     desc="💫 Augmentasi") as pbar:
                
                # Process dengan threading
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = []
                    
                    for img_path in image_files:
                        # Tentukan path file
                        label_path = (input_path / 'labels' / 
                                    img_path.stem).with_suffix('.txt')
                        
                        future = executor.submit(
                            self._process_and_save,
                            img_path,
                            label_path,
                            augmentation_type,
                            pbar
                        )
                        futures.append(future)
                        
                    # Collect results
                    for future in futures:
                        success = future.result()
                        if success:
                            stats['augmented'] += 3  # 3 variasi per gambar
                        else:
                            stats['failed'] += 1
                            
            self.logger.success(
                f"✨ Augmentasi selesai!\n"
                f"📊 Statistik:\n"
                f"   Original: {stats['original']} gambar\n"
                f"   Augmented: {stats['augmented']} gambar\n"
                f"   Gagal: {stats['failed']} gambar"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Augmentasi gagal: {str(e)}")
            raise e
            
    def _process_and_save(
        self,
        img_path: Path,
        label_path: Path,
        augmentation_type: str,
        pbar: tqdm
    ) -> bool:
        """Proses dan simpan hasil augmentasi satu gambar"""
        try:
            # Augmentasi
            aug_images, aug_bboxes = self.augment_image(
                img_path,
                label_path,
                augmentation_type
            )
            
            # Simpan hasil
            for i, (aug_img, aug_bbox) in enumerate(zip(aug_images, aug_bboxes)):
                # Generate nama file
                suffix = f"_{augmentation_type}_{i+1}"
                aug_img_path = self.output_dir / 'images' / \
                              f"{img_path.stem}{suffix}.jpg"
                aug_label_path = self.output_dir / 'labels' / \
                                f"{img_path.stem}{suffix}.txt"
                
                # Buat direktori jika belum ada
                aug_img_path.parent.mkdir(exist_ok=True)
                aug_label_path.parent.mkdir(exist_ok=True)
                
                # Simpan gambar
                cv2.imwrite(
                    str(aug_img_path),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                )
                
                # Simpan label
                if len(aug_bbox) > 0:
                    np.savetxt(aug_label_path, aug_bbox, fmt='%.6f')
                    
            pbar.update(1)
            return True
            
        except Exception as e:
            self.logger.warning(
                f"⚠️ Gagal mengaugmentasi {img_path.name}: {str(e)}"
            )
            pbar.update(1)
            return False