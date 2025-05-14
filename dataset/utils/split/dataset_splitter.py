"""
File: smartcash/dataset/utils/split/dataset_splitter.py
Deskripsi: Utilitas untuk memecah dataset menjadi train/valid/test
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_SPLIT_RATIOS, DEFAULT_RANDOM_SEED


class DatasetSplitter:
    """Utilitas untuk memecah dataset menjadi train/valid/test."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi DatasetSplitter.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("dataset_splitter")
        
        self.logger.info(f"✂️ DatasetSplitter diinisialisasi dengan data_dir: {self.data_dir}")
    
    def split_dataset(
        self, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.2, 
        test_ratio: float = 0.1,
        random_seed: int = DEFAULT_RANDOM_SEED, 
        stratify_by_class: bool = True, 
        create_symlinks: bool = False,
        source_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, int]:
        """
        Pecah dataset menjadi train/val/test berdasarkan rasio yang diberikan.
        
        Args:
            train_ratio: Rasio untuk split train
            val_ratio: Rasio untuk split validation
            test_ratio: Rasio untuk split test
            random_seed: Seed untuk random
            stratify_by_class: Apakah stratifikasi berdasarkan kelas
            create_symlinks: Apakah menggunakan symlink alih-alih menyalin file
            source_dir: Direktori sumber (opsional)
            
        Returns:
            Dictionary berisi jumlah file per split
        """
        # Validasi rasio
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError(f"Rasio harus berjumlah 1.0, didapat: {train_ratio + val_ratio + test_ratio}")
        
        # Set random seed
        random.seed(random_seed)
        
        # Tentukan direktori sumber dan output
        source_dir = Path(source_dir) if source_dir else self.data_dir
        
        # Cek struktur data
        data_structure = self._detect_data_structure(source_dir)
        
        if data_structure == 'flat':
            return self._split_flat_dataset(
                source_dir, 
                train_ratio, 
                val_ratio, 
                test_ratio, 
                stratify_by_class, 
                create_symlinks
            )
        elif data_structure == 'split':
            self.logger.info("📁 Dataset sudah terbagi dalam train/valid/test, menggunakan struktur yang ada")
            return self._count_existing_splits(source_dir)
        else:
            raise ValueError(f"Struktur dataset tidak dikenal: {data_structure}")
    
    def _detect_data_structure(self, directory: Path) -> str:
        """
        Deteksi struktur dataset yang ada.
        
        Args:
            directory: Direktori dataset
            
        Returns:
            Jenis struktur: 'split', 'flat', atau ValueError
        """
        # Cek apakah ada direktori train/valid/test
        has_splits = all([(directory / split).exists() for split in DEFAULT_SPLITS])
        
        # Cek apakah ada direktori images/labels di root
        has_flat = (directory / 'images').exists() and (directory / 'labels').exists()
        
        if has_splits:
            # Cek apakah struktur split valid (memiliki images dan labels)
            is_valid_split = all([
                (directory / split / 'images').exists() and 
                (directory / split / 'labels').exists()
                for split in DEFAULT_SPLITS
            ])
            if is_valid_split:
                return 'split'
        
        if has_flat:
            return 'flat'
            
        # Jika tidak ada struktur yang cocok, periksa isi direktori
        if list(directory.glob('*.jpg')) or list(directory.glob('*.txt')):
            self.logger.warning("⚠️ Dataset memiliki struktur tidak standar, mencoba sebagai flat")
            return 'flat'
            
        raise ValueError(f"Struktur dataset tidak dikenali di: {directory}")
    
    def _split_flat_dataset(
        self, 
        source_dir: Path,
        train_ratio: float, 
        val_ratio: float, 
        test_ratio: float,
        stratify_by_class: bool, 
        create_symlinks: bool
    ) -> Dict[str, int]:
        """
        Pecah dataset dengan struktur flat.
        
        Args:
            source_dir: Direktori sumber
            train_ratio: Rasio untuk split train
            val_ratio: Rasio untuk split validation
            test_ratio: Rasio untuk split test
            stratify_by_class: Apakah stratifikasi berdasarkan kelas
            create_symlinks: Apakah menggunakan symlink alih-alih menyalin file
            
        Returns:
            Dictionary berisi jumlah file per split
        """
        self.logger.info(f"📊 Memecah dataset dengan rasio: {train_ratio}/{val_ratio}/{test_ratio}")
        
        # Buat direktori output jika belum ada
        for split in DEFAULT_SPLITS:
            for subdir in ['images', 'labels']:
                (self.data_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Cari semua file gambar dan filter yang memiliki label
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            if (source_dir / 'images').exists():
                image_files.extend(list((source_dir / 'images').glob(ext)))
            else:
                image_files.extend(list(source_dir.glob(ext)))
        
        valid_files = []
        for img_path in image_files:
            label_path = (source_dir / 'labels' / f"{img_path.stem}.txt") if (source_dir / 'labels').exists() else source_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_files.append((img_path, label_path))
                
        if not valid_files:
            self.logger.error("❌ Tidak ada file valid dengan pasangan gambar/label")
            return {split: 0 for split in DEFAULT_SPLITS}
            
        self.logger.info(f"🔍 Ditemukan {len(valid_files)} file valid dengan pasangan gambar/label")
        
        # Stratifikasi berdasarkan kelas jika diminta
        files_by_class = {}
        if stratify_by_class:
            from smartcash.dataset.utils.dataset_utils import DatasetUtils
            utils = DatasetUtils(self.config, str(self.data_dir), self.logger)
            
            for img_path, label_path in valid_files:
                bbox_data = utils.parse_yolo_label(label_path)
                classes = [box['class_id'] for box in bbox_data]
                main_class = classes[0] if classes else 'unknown'
                
                if main_class not in files_by_class:
                    files_by_class[main_class] = []
                files_by_class[main_class].append((img_path, label_path))
                    
            # Log distribusi kelas
            self.logger.info(f"📊 Distribusi kelas:")
            for cls, files in files_by_class.items():
                class_name = utils.get_class_name(cls) if cls != 'unknown' else "Unknown"
                self.logger.info(f"   • Kelas {class_name}: {len(files)} sampel")
        else:
            files_by_class = {'all': valid_files}
        
        # Lakukan pemecahan
        train_files, val_files, test_files = [], [], []
        
        for _, files in files_by_class.items():
            random.shuffle(files)
            n_total = len(files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train + n_val])
            test_files.extend(files[n_train + n_val:])
        
        # Log jumlah file per split
        self.logger.info(
            f"📊 Pembagian dataset:\n"
            f"   • Train: {len(train_files)} sampel\n"
            f"   • Valid: {len(val_files)} sampel\n"
            f"   • Test: {len(test_files)} sampel"
        )
        
        # Salin atau buat symlink file
        self._copy_files('train', train_files, create_symlinks)
        self._copy_files('valid', val_files, create_symlinks)
        self._copy_files('test', test_files, create_symlinks)
        
        # Return statistik
        return {
            'train': len(train_files),
            'valid': len(val_files),
            'test': len(test_files)
        }
    
    def _copy_files(self, split: str, files: List[Tuple[Path, Path]], use_symlinks: bool) -> None:
        """
        Salin file ke direktori split.
        
        Args:
            split: Nama split
            files: List pasangan file (gambar, label)
            use_symlinks: Apakah menggunakan symlink
        """
        self.logger.info(f"📂 Memindahkan {len(files)} file ke {split}...")
        
        with tqdm(files, desc=f"Copying to {split}") as pbar:
            for img_path, label_path in pbar:
                target_img = self.data_dir / split / 'images' / img_path.name
                target_label = self.data_dir / split / 'labels' / label_path.name
                
                try:
                    if use_symlinks:
                        if not target_img.exists():
                            target_img.symlink_to(img_path.resolve())
                        if not target_label.exists():
                            target_label.symlink_to(label_path.resolve())
                    else:
                        if not target_img.exists():
                            shutil.copy2(img_path, target_img)
                        if not target_label.exists():
                            shutil.copy2(label_path, target_label)
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memindahkan {img_path.name}: {str(e)}")
    
    def _count_existing_splits(self, directory: Path) -> Dict[str, int]:
        """
        Hitung jumlah file dalam split yang sudah ada.
        
        Args:
            directory: Direktori dataset
            
        Returns:
            Dictionary berisi jumlah file per split
        """
        counts = {}
        
        for split in DEFAULT_SPLITS:
            images_dir = directory / split / 'images'
            if images_dir.exists():
                image_count = sum(1 for _ in images_dir.glob('*.jpg')) + \
                              sum(1 for _ in images_dir.glob('*.jpeg')) + \
                              sum(1 for _ in images_dir.glob('*.png'))
                counts[split] = image_count
            else:
                counts[split] = 0
                
        self.logger.info(
            f"📊 Statistik split dataset yang ada:\n"
            f"   • Train: {counts.get('train', 0)} gambar\n"
            f"   • Valid: {counts.get('valid', 0)} gambar\n"
            f"   • Test: {counts.get('test', 0)} gambar"
        )
        return counts