"""
File: smartcash/utils/preprocessing.py
Author: Alfrida Sabar
Deskripsi: Implementasi pipeline preprocessing untuk dataset deteksi mata uang dengan dukungan 
           parallelism, caching, dan monitoring progress.
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union
import concurrent.futures
import shutil
import time
import random
from dataclasses import dataclass

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.enhanced_cache import EnhancedCache

@dataclass
class PreprocessingConfig:
    """Konfigurasi untuk pipeline preprocessing."""
    input_dir: str
    output_dir: str
    img_size: Tuple[int, int] = (640, 640)
    split_ratio: Dict[str, float] = None
    cache_dir: str = ".cache/smartcash"
    num_workers: int = 4
    augmentation_enabled: bool = True
    normalize_enabled: bool = True

class PreprocessingPipeline:
    """
    Pipeline untuk preprocessing dataset deteksi mata uang dengan dukungan
    multi-processing, augmentasi, dan monitoring progress.
    """
    
    def __init__(
        self,
        config: Union[Dict, PreprocessingConfig],
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi pipeline preprocessing.
        
        Args:
            config: Konfigurasi preprocessing (dict atau PreprocessingConfig)
            logger: Logger untuk mencatat progress
        """
        # Setup config
        if isinstance(config, dict):
            self.config = PreprocessingConfig(
                input_dir=config.get('input_dir', 'data/raw'),
                output_dir=config.get('output_dir', 'data'),
                img_size=config.get('img_size', (640, 640)),
                split_ratio=config.get('split_ratio', {'train': 0.8, 'valid': 0.1, 'test': 0.1}),
                cache_dir=config.get('cache_dir', '.cache/smartcash'),
                num_workers=config.get('num_workers', 4),
                augmentation_enabled=config.get('augmentation_enabled', True),
                normalize_enabled=config.get('normalize_enabled', True)
            )
        else:
            self.config = config
            
        # Setup logger
        self.logger = logger or SmartCashLogger("preprocessing")
        
        # Setup cache
        self.cache = EnhancedCache(
            cache_dir=self.config.cache_dir,
            logger=self.logger
        )
        
        # Setup transformasi dasar
        self.transform = A.Compose([
            A.Resize(height=self.config.img_size[1], width=self.config.img_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.config.normalize_enabled else A.NoOp(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Setup transformasi augmentasi
        self.aug_transform = A.Compose([
            A.RandomResizedCrop(height=self.config.img_size[1], width=self.config.img_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.config.normalize_enabled else A.NoOp(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def run_pipeline(self) -> Dict:
        """
        Jalankan pipeline preprocessing lengkap.
        
        Returns:
            Dict berisi statistik hasil preprocessing
        """
        self.logger.start("🔄 Memulai pipeline preprocessing...")
        
        # 1. Pastikan direktori input dan output ada
        self._validate_directories()
        
        # 2. Split dataset dan distribusikan ke direktori output
        dataset_split = self.split_dataset()
        
        # 3. Preprocessing data (resize, normalize)
        preprocess_stats = self.preprocess_data()
        
        # 4. Lakukan augmentasi data (jika diaktifkan)
        aug_stats = {}
        if self.config.augmentation_enabled:
            aug_stats = self.augment_data()
            
        # Gabungkan statistik
        stats = {
            'split': dataset_split,
            'preprocessing': preprocess_stats,
            'augmentation': aug_stats
        }
            
        self.logger.success("✅ Pipeline preprocessing selesai!")
        return stats
    
    def _validate_directories(self) -> None:
        """Validasi dan buat direktori yang diperlukan."""
        # Cek input direktori
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            self.logger.warning(f"⚠️ Direktori input {input_dir} tidak ditemukan, membuat direktori baru")
            input_dir.mkdir(parents=True, exist_ok=True)
            
        # Buat direktori output
        base_output = Path(self.config.output_dir)
        for split in ['train', 'valid', 'test']:
            for subdir in ['images', 'labels']:
                (base_output / split / subdir).mkdir(parents=True, exist_ok=True)
                
        self.logger.info(f"✅ Direktori tervalidasi: {self.config.output_dir}")
    
    def split_dataset(self) -> Dict:
        """
        Split dataset ke train, validation, dan test set.
        
        Returns:
            Dict berisi statistik split dataset
        """
        self.logger.start("📊 Memulai split dataset...")
        
        input_dir = Path(self.config.input_dir)
        output_dir = Path(self.config.output_dir)
        
        # Cari semua file gambar dalam direktori input
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(input_dir.glob(f"images/{ext}")))
            
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ditemukan gambar di {input_dir}/images")
            return {"total_images": 0, "train": 0, "valid": 0, "test": 0}
            
        # Acak dataset untuk memastikan distribusi yang baik
        random.shuffle(image_files)
        
        # Hitung jumlah gambar untuk setiap split
        total_images = len(image_files)
        split_counts = {
            'train': int(self.config.split_ratio['train'] * total_images),
            'valid': int(self.config.split_ratio['valid'] * total_images),
            'test': int(self.config.split_ratio['test'] * total_images)
        }
        
        # Koreksi total jika tidak sama dengan total_images
        diff = total_images - sum(split_counts.values())
        split_counts['train'] += diff
        
        # Split dataset
        splits = {}
        start_idx = 0
        for split, count in split_counts.items():
            splits[split] = image_files[start_idx:start_idx + count]
            start_idx += count
            
        # Salin file ke direktori output
        stats = {"total_images": total_images}
        
        for split, files in splits.items():
            stats[split] = len(files)
            self.logger.info(f"🔄 Menyalin {len(files)} file ke split {split}...")
            
            for img_file in tqdm(files, desc=f"Split {split}", unit="file"):
                # Salin gambar
                dest_img = output_dir / split / "images" / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Salin label jika ada
                label_file = input_dir / "labels" / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = output_dir / split / "labels" / f"{img_file.stem}.txt"
                    shutil.copy2(label_file, dest_label)
                    
        self.logger.success(
            f"✅ Split dataset selesai:\n"
            f"   • Total: {total_images} gambar\n"
            f"   • Train: {stats['train']} gambar\n"
            f"   • Valid: {stats['valid']} gambar\n"
            f"   • Test: {stats['test']} gambar"
        )
        
        return stats
    
    def preprocess_data(self) -> Dict:
        """
        Lakukan preprocessing pada dataset (resize, normalize).
        
        Returns:
            Dict berisi statistik preprocessing
        """
        self.logger.start("🔄 Memulai preprocessing dataset...")
        
        output_dir = Path(self.config.output_dir)
        
        # Cari semua file gambar dalam direktori output
        stats = {"total": 0}
        for split in ['train', 'valid', 'test']:
            image_dir = output_dir / split / "images"
            label_dir = output_dir / split / "labels"
            
            # Dapatkan semua file gambar
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(image_dir.glob(ext)))
            
            if not image_files:
                stats[split] = {"processed": 0, "skipped": 0}
                continue
                
            self.logger.info(f"🔄 Preprocessing {len(image_files)} gambar di split {split}...")
            
            # Track statistik
            split_stats = {"processed": 0, "skipped": 0}
            
            # Proses gambar secara paralel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                
                for img_file in image_files:
                    # Jika file sudah dalam cache, skip
                    cache_key = f"preprocess_{split}_{img_file.name}"
                    if self.cache.exists(cache_key):
                        split_stats["skipped"] += 1
                        continue
                        
                    futures.append(
                        executor.submit(
                            self._preprocess_single_image,
                            img_file=img_file,
                            label_dir=label_dir,
                            cache_key=cache_key
                        )
                    )
                
                # Collect results with progress
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                            desc=f"Preprocessing {split}", unit="gambar"):
                    split_stats["processed"] += 1
            
            stats[split] = split_stats
            stats["total"] += split_stats["processed"] + split_stats["skipped"]
                
        self.logger.success(f"✅ Preprocessing selesai: {stats['total']} gambar")
        return stats
    
    def _preprocess_single_image(self, img_file: Path, label_dir: Path, cache_key: str) -> bool:
        """
        Preprocess satu gambar dan labelnya.
        
        Args:
            img_file: Path ke file gambar
            label_dir: Path ke direktori label
            cache_key: Key untuk cache
            
        Returns:
            Boolean yang menunjukkan keberhasilan preprocessing
        """
        try:
            # Baca gambar
            img = cv2.imread(str(img_file))
            if img is None:
                return False
                
            # Baca label jika ada
            label_file = label_dir / f"{img_file.stem}.txt"
            bboxes = []
            class_labels = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Apply transformasi
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Simpan hasil transformasi
            cv2.imwrite(str(img_file), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            
            # Jika ada bounding box yang berubah, update label
            if label_file.exists() and transformed['bboxes']:
                with open(label_file, 'w') as f:
                    for i, bbox in enumerate(transformed['bboxes']):
                        class_id = transformed['class_labels'][i]
                        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
            # Simpan ke cache
            self.cache.set(cache_key, True)
            return True
            
        except Exception as e:
            # Log error tanpa memblokir thread
            print(f"⚠️ Error pada {img_file.name}: {str(e)}")
            return False
    
    def augment_data(self) -> Dict:
        """
        Lakukan augmentasi pada training set untuk meningkatkan variasi data.
        
        Returns:
            Dict berisi statistik augmentasi
        """
        self.logger.start("🎨 Memulai augmentasi data...")
        
        output_dir = Path(self.config.output_dir)
        train_img_dir = output_dir / "train" / "images"
        train_label_dir = output_dir / "train" / "labels"
        
        # Dapatkan semua file gambar training
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(train_img_dir.glob(ext)))
            
        # Filter hanya yang memiliki label
        image_files = [img for img in image_files if (train_label_dir / f"{img.stem}.txt").exists()]
        
        if not image_files:
            self.logger.warning("⚠️ Tidak ada gambar berlabel untuk augmentasi")
            return {"total": 0, "augmented": 0, "skipped": 0}
            
        # Track statistik
        stats = {"total": len(image_files), "augmented": 0, "skipped": 0}
        
        # Lakukan augmentasi pada subset gambar (50% dari total)
        subset_size = len(image_files) // 2
        subset = random.sample(image_files, subset_size)
        
        self.logger.info(f"🎨 Augmentasi {subset_size} gambar...")
        
        # Proses secara paralel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for img_file in subset:
                # Cek cache
                cache_key = f"augment_{img_file.name}"
                if self.cache.exists(cache_key):
                    stats["skipped"] += 1
                    continue
                    
                futures.append(
                    executor.submit(
                        self._augment_single_image,
                        img_file=img_file,
                        label_file=train_label_dir / f"{img_file.stem}.txt",
                        img_dir=train_img_dir,
                        label_dir=train_label_dir,
                        cache_key=cache_key
                    )
                )
            
            # Collect results with progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                             desc="Augmentasi", unit="gambar"):
                if future.result():
                    stats["augmented"] += 1
        
        self.logger.success(f"✅ Augmentasi selesai: {stats['augmented']} gambar baru")
        return stats
    
    def _augment_single_image(self, img_file: Path, label_file: Path, 
                            img_dir: Path, label_dir: Path, cache_key: str) -> bool:
        """
        Augmentasi satu gambar dan labelnya.
        
        Args:
            img_file: Path ke file gambar
            label_file: Path ke file label
            img_dir: Path ke direktori output gambar
            label_dir: Path ke direktori output label
            cache_key: Key untuk cache
            
        Returns:
            Boolean yang menunjukkan keberhasilan augmentasi
        """
        try:
            # Baca gambar
            img = cv2.imread(str(img_file))
            if img is None:
                return False
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Baca label
            bboxes = []
            class_labels = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Apply augmentasi
            transformed = self.aug_transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Nama file baru untuk hasil augmentasi
            aug_img_name = f"aug_{img_file.stem}_{int(time.time() * 1000)}{img_file.suffix}"
            aug_label_name = f"aug_{img_file.stem}_{int(time.time() * 1000)}.txt"
            
            # Simpan hasil augmentasi
            aug_img_path = img_dir / aug_img_name
            aug_label_path = label_dir / aug_label_name
            
            # Simpan gambar
            cv2.imwrite(str(aug_img_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            
            # Simpan label
            with open(aug_label_path, 'w') as f:
                for i, bbox in enumerate(transformed['bboxes']):
                    class_id = transformed['class_labels'][i]
                    f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
            # Simpan ke cache
            self.cache.set(cache_key, True)
            return True
            
        except Exception as e:
            # Log error tanpa memblokir thread
            print(f"⚠️ Error augmentasi {img_file.name}: {str(e)}")
            return False