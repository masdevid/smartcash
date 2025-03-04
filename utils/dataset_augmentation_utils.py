# File: smartcash/utils/dataset_augmentation_utils.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk preprocessing dan augmentasi dataset dengan dukungan progres bar dan pengaturan rasio split

import os
import cv2
import numpy as np
import albumentations as A
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
from typing import Dict, List, Optional, Tuple, Union

class DatasetProcessor:
    """Kelas untuk mengelola preprocessing dan augmentasi dataset SmartCash."""
    
    def __init__(
        self, 
        data_dir: str = "data",
        output_dir: Optional[str] = None,
        logger = None
    ):
        """
        Inisialisasi processor dataset.
        
        Args:
            data_dir: Direktori data utama
            output_dir: Direktori output (jika None, gunakan data_dir)
            logger: Logger untuk output (opsional)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Setup logger functions
        if logger:
            self.log_info = logger.info
            self.log_error = logger.error
            self.log_success = logger.success
            self.log_warning = logger.warning
        else:
            # Default to print jika tidak ada logger
            self.log_info = lambda msg: print(f"ℹ️ {msg}")
            self.log_error = lambda msg: print(f"❌ {msg}")
            self.log_success = lambda msg: print(f"✅ {msg}")
            self.log_warning = lambda msg: print(f"⚠️ {msg}")
        
        # Setup augmentation pipelines
        self._setup_augmentation_pipelines()
    
    def _setup_augmentation_pipelines(self):
        """Setup transformasi augmentasi untuk berbagai kondisi."""
        
        # Pipeline posisi - variasi posisi/orientasi uang
        self.position_aug = A.Compose([
            A.SafeRotate(limit=30, p=0.7),
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
        
        # Pipeline pencahayaan - variasi cahaya dan kontras
        self.lighting_aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),
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
        
        # Pipeline kombinasi - gabungan posisi dan pencahayaan
        self.combined_aug = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=30, p=0.7),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5
                )
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomShadow(p=0.5),
                A.HueSaturationValue(p=0.5)
            ], p=0.7),
            A.HorizontalFlip(p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def augment_image(
        self,
        image_path: Path,
        label_path: Optional[Path] = None,
        augmentation_type: str = 'combined',
        output_prefix: str = 'aug'
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path]]:
        """
        Augmentasi satu gambar dengan label.
        
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (opsional)
            augmentation_type: Tipe augmentasi ('position', 'lighting', atau 'combined')
            output_prefix: Prefix untuk nama file output
            
        Returns:
            Tuple of (augmented_images, augmented_labels, output_paths)
        """
        # Baca gambar
        image = cv2.imread(str(image_path))
        if image is None:
            self.log_error(f"Tidak dapat membaca gambar: {image_path}")
            return [], [], []
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Baca label jika ada
        bboxes = []
        class_labels = []
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id, x, y, w, h
                        cls = float(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        bboxes.append(bbox)
                        class_labels.append(cls)
        
        # Pilih pipeline augmentasi
        if augmentation_type == 'position':
            augmentor = self.position_aug
        elif augmentation_type == 'lighting':
            augmentor = self.lighting_aug
        else:
            augmentor = self.combined_aug
        
        # Generate 3 variasi
        augmented_images = []
        augmented_bboxes = []
        output_paths = []
        
        for i in range(3):
            # Apply augmentation
            try:
                augmented = augmentor(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                augmented_images.append(augmented['image'])
                
                # Generate output paths
                suffix = f"{output_prefix}_{augmentation_type}_{i+1}"
                img_output_path = image_path.parent / f"{image_path.stem}_{suffix}{image_path.suffix}"
                
                output_paths.append(img_output_path)
                
                # Save bbox if needed
                if bboxes:
                    augmented_bboxes.append(
                        list(zip(augmented['class_labels'], 
                                augmented['bboxes']))
                    )
            except Exception as e:
                self.log_error(f"Augmentasi gagal untuk {image_path.name}: {str(e)}")
                continue
                
        return augmented_images, augmented_bboxes, output_paths
    
    def save_augmented_data(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[float, List[float]]],
        image_path: Path,
        labels_dir: Path
    ) -> bool:
        """
        Simpan hasil augmentasi ke file.
        
        Args:
            image: Gambar hasil augmentasi
            bboxes: Bounding box hasil augmentasi [(class_id, [x, y, w, h]), ...]
            image_path: Path untuk menyimpan gambar
            labels_dir: Direktori untuk menyimpan label
            
        Returns:
            Boolean sukses/gagal
        """
        try:
            # Simpan gambar
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            # Simpan label jika ada
            if bboxes:
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for cls, bbox in bboxes:
                        line = f"{int(cls)} {' '.join(map(str, bbox))}\n"
                        f.write(line)
                        
            return True
        except Exception as e:
            self.log_error(f"Gagal menyimpan hasil augmentasi: {str(e)}")
            return False
    
    def augment_dataset(
        self,
        split: str = 'train',
        augmentation_type: str = 'combined',
        num_workers: int = 4,
        output_prefix: str = 'aug'
    ) -> Dict[str, int]:
        """
        Augmentasi dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            augmentation_type: Tipe augmentasi ('position', 'lighting', atau 'combined')
            num_workers: Jumlah worker untuk multiprocessing
            output_prefix: Prefix untuk nama file output
            
        Returns:
            Dict statistik augmentasi
        """
        # Setup direktori
        input_dir = self.data_dir / split
        output_dir = self.output_dir / split
        
        images_dir = input_dir / 'images'
        labels_dir = input_dir / 'labels'
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        
        # Pastikan direktori output ada
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Periksa file gambar yang tersedia
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        
        if not image_files:
            self.log_warning(f"Tidak ada gambar ditemukan di {images_dir}")
            return {'processed': 0, 'augmented': 0, 'errors': 0}
        
        self.log_info(f"Menemukan {len(image_files)} gambar di {split}")
        
        # Statistik
        stats = {'processed': 0, 'augmented': 0, 'errors': 0}
        
        # Fungsi untuk memproses satu gambar
        def process_image(img_path):
            local_stats = {'augmented': 0, 'errors': 0}
            
            # Cari label yang sesuai
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Augmentasi
            aug_images, aug_bboxes, out_paths = self.augment_image(
                img_path,
                label_path if label_path.exists() else None,
                augmentation_type,
                output_prefix
            )
            
            # Simpan hasil
            for idx, (img, out_path) in enumerate(zip(aug_images, out_paths)):
                # Ubah path ke direktori output
                save_path = output_images_dir / out_path.name
                
                # Jika ada bboxes, ambil yang sesuai dengan gambar ini
                bboxes_to_save = aug_bboxes[idx] if aug_bboxes and idx < len(aug_bboxes) else []
                
                # Simpan hasil
                if self.save_augmented_data(img, bboxes_to_save, save_path, output_labels_dir):
                    local_stats['augmented'] += 1
                else:
                    local_stats['errors'] += 1
            
            return local_stats
        
        # Proses menggunakan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc=f"Augmentasi {split}"
            ))
            
        # Agregasi hasil
        stats['processed'] = len(image_files)
        for result in results:
            stats['augmented'] += result['augmented']
            stats['errors'] += result['errors']
            
        self.log_success(
            f"Augmentasi {split} selesai: "
            f"{stats['augmented']} gambar dihasilkan, "
            f"{stats['errors']} error"
        )
        
        return stats
    
    def augment_all_splits(
        self, 
        augmentation_type: str = 'combined',
        splits: List[str] = ['train'],
        num_workers: int = 4,
        output_prefix: str = 'aug'
    ) -> Dict[str, Dict[str, int]]:
        """
        Augmentasi semua split dataset.
        
        Args:
            augmentation_type: Tipe augmentasi
            splits: List split yang akan diaugmentasi
            num_workers: Jumlah worker
            output_prefix: Prefix untuk nama file
            
        Returns:
            Dict statistik per split
        """
        all_stats = {}
        
        for split in splits:
            self.log_info(f"Memulai augmentasi {split}...")
            stats = self.augment_dataset(
                split=split,
                augmentation_type=augmentation_type,
                num_workers=num_workers,
                output_prefix=output_prefix
            )
            all_stats[split] = stats
            
        return all_stats
    
    def clean_augmented_data(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        prefix: str = 'aug'
    ) -> Dict[str, int]:
        """
        Bersihkan file hasil augmentasi.
        
        Args:
            splits: List split yang akan dibersihkan
            prefix: Prefix file augmentasi
            
        Returns:
            Dict statistik pembersihan
        """
        stats = {'removed_images': 0, 'removed_labels': 0}
        
        for split in splits:
            # Setup direktori
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            # Cari file augmentasi gambar
            aug_images = list(images_dir.glob(f'*{prefix}*.*'))
            
            # Hapus gambar
            for img_path in tqdm(aug_images, desc=f"Membersihkan {split} images"):
                try:
                    img_path.unlink()
                    stats['removed_images'] += 1
                except Exception as e:
                    self.log_error(f"Gagal menghapus {img_path}: {str(e)}")
            
            # Hapus label yang sesuai
            for img_path in aug_images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    try:
                        label_path.unlink()
                        stats['removed_labels'] += 1
                    except Exception as e:
                        self.log_error(f"Gagal menghapus {label_path}: {str(e)}")
        
        self.log_success(
            f"Pembersihan selesai: "
            f"{stats['removed_images']} gambar, "
            f"{stats['removed_labels']} label"
        )
        
        return stats
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik dataset.
        
        Returns:
            Dict statistik per split
        """
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                stats[split] = {'images': 0, 'labels': 0, 'augmented': 0}
                continue
                
            # Hitung jumlah file
            all_images = list(images_dir.glob('*.*'))
            all_labels = list(labels_dir.glob('*.txt'))
            
            # Hitung jumlah file augmentasi
            aug_images = [img for img in all_images if 'aug' in img.name]
            
            stats[split] = {
                'images': len(all_images),
                'labels': len(all_labels),
                'augmented': len(aug_images),
                'original': len(all_images) - len(aug_images)
            }
            
        return stats
    
    def split_dataset(
        self, 
        train_ratio: float = 0.7, 
        valid_ratio: float = 0.15, 
        test_ratio: float = 0.15
    ) -> Dict[str, int]:
        """
        Split dataset menjadi train, valid, dan test.
        
        Args:
            train_ratio: Rasio data training
            valid_ratio: Rasio data validasi
            test_ratio: Rasio data testing
            
        Returns:
            Dict statistik split
        """
        # Validasi rasio
        total_ratio = train_ratio + valid_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            self.log_error(f"Total rasio harus 1.0, sekarang {total_ratio}")
            train_ratio = 0.7
            valid_ratio = 0.15
            test_ratio = 0.15
            self.log_warning(f"Menggunakan rasio default: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")
        
        # Cari semua file yang tersedia di direktori data
        all_data_dir = self.data_dir / 'all_data'
        if not all_data_dir.exists() or not (all_data_dir / 'images').exists():
            self.log_error(f"Direktori all_data/images tidak ditemukan. Pastikan dataset tersedia.")
            return {'train': 0, 'valid': 0, 'test': 0}
        
        images_dir = all_data_dir / 'images'
        labels_dir = all_data_dir / 'labels'
        
        # Cek ketersediaan data
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        if not image_files:
            self.log_error(f"Tidak ada gambar ditemukan di {images_dir}")
            return {'train': 0, 'valid': 0, 'test': 0}
        
        # Acak urutan file
        random.shuffle(image_files)
        
        # Hitung jumlah file untuk masing-masing split
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        n_test = n_total - n_train - n_valid
        
        # Split file
        train_files = image_files[:n_train]
        valid_files = image_files[n_train:n_train+n_valid]
        test_files = image_files[n_train+n_valid:]
        
        # Buat direktori untuk masing-masing split
        for split in ['train', 'valid', 'test']:
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Fungsi untuk memindahkan file
        def move_files(files, split):
            moved_count = 0
            for img_path in tqdm(files, desc=f"Memindahkan {split}"):
                # Path tujuan
                dst_img_path = self.data_dir / split / 'images' / img_path.name
                
                # Path label
                label_path = labels_dir / f"{img_path.stem}.txt"
                dst_label_path = self.data_dir / split / 'labels' / f"{img_path.stem}.txt"
                
                # Copy gambar
                try:
                    shutil.copy2(img_path, dst_img_path)
                    
                    # Copy label jika ada
                    if label_path.exists():
                        shutil.copy2(label_path, dst_label_path)
                        
                    moved_count += 1
                except Exception as e:
                    self.log_error(f"Gagal memindahkan {img_path}: {str(e)}")
                    
            return moved_count
        
        # Pindahkan file
        stats = {
            'train': move_files(train_files, 'train'),
            'valid': move_files(valid_files, 'valid'),
            'test': move_files(test_files, 'test')
        }
        
        self.log_success(
            f"Split dataset selesai:\n"
            f"Train: {stats['train']} file\n"
            f"Valid: {stats['valid']} file\n"
            f"Test: {stats['test']} file"
        )
        
        return stats