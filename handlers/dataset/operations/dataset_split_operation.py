# File: smartcash/handlers/dataset/operations/dataset_split_operation.py
# Author: Alfrida Sabar
# Deskripsi: Operasi untuk memecah dataset menjadi train, validation, dan test

import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger


class DatasetSplitOperation:
    """Operasi untuk memecah dataset menjadi train, validation, dan test."""
    
    def __init__(self, data_dir: str, output_dir: Optional[str] = None, logger: Optional[SmartCashLogger] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.logger = logger or SmartCashLogger(__name__)
        self.logger.info(f"🔧 DatasetSplitOperation diinisialisasi: {self.data_dir}")
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                      random_seed: int = 42, stratify_by_class: bool = True, create_symlinks: bool = False) -> Dict[str, int]:
        """Pecah dataset menjadi train/val/test berdasarkan rasio yang diberikan."""
        # Validasi rasio
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError(f"Rasio harus berjumlah 1.0, didapat: {train_ratio + val_ratio + test_ratio}")
            
        random.seed(random_seed)
        data_structure = self._detect_data_structure()
        
        if data_structure == 'flat':
            return self._split_flat_dataset(train_ratio, val_ratio, test_ratio, stratify_by_class, create_symlinks)
        elif data_structure == 'split':
            self.logger.info("📁 Dataset sudah terbagi dalam train/valid/test, menggunakan struktur yang ada")
            return self._count_existing_splits()
        else:
            raise ValueError(f"Struktur dataset tidak dikenal: {data_structure}")
    
    def _detect_data_structure(self) -> str:
        """Deteksi struktur dataset yang ada."""
        # Cek apakah ada direktori train/valid/test
        has_splits = all([(self.data_dir / split).exists() for split in ['train', 'valid', 'test']])
        
        # Cek apakah ada direktori images/labels di root
        has_flat = (self.data_dir / 'images').exists() and (self.data_dir / 'labels').exists()
        
        if has_splits:
            # Cek apakah struktur split valid (memiliki images dan labels)
            is_valid_split = all([
                (self.data_dir / split / 'images').exists() and 
                (self.data_dir / split / 'labels').exists()
                for split in ['train', 'valid', 'test']
            ])
            if is_valid_split:
                return 'split'
        
        if has_flat:
            return 'flat'
            
        # Jika tidak ada struktur yang cocok, periksa isi direktori
        if list(self.data_dir.glob('*.jpg')) or list(self.data_dir.glob('*.txt')):
            self.logger.warning("⚠️ Dataset memiliki struktur tidak standar, mencoba sebagai flat")
            return 'flat'
            
        raise ValueError(f"Struktur dataset tidak dikenali di: {self.data_dir}")
    
    def _split_flat_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float,
                           stratify_by_class: bool, create_symlinks: bool) -> Dict[str, int]:
        """Pecah dataset dengan struktur flat."""
        self.logger.info(f"📊 Memecah dataset dengan rasio: {train_ratio}/{val_ratio}/{test_ratio}")
        
        # Buat direktori output jika belum ada
        for split in ['train', 'valid', 'test']:
            for subdir in ['images', 'labels']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Cari semua file gambar dan filter yang memiliki label
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            if (self.data_dir / 'images').exists():
                image_files.extend(list((self.data_dir / 'images').glob(ext)))
            else:
                image_files.extend(list(self.data_dir.glob(ext)))
        
        valid_files = []
        for img_path in image_files:
            label_path = (self.data_dir / 'labels' / f"{img_path.stem}.txt") if (self.data_dir / 'labels').exists() else self.data_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_files.append((img_path, label_path))
                
        if not valid_files:
            self.logger.error("❌ Tidak ada file valid dengan pasangan gambar/label")
            return {'train': 0, 'valid': 0, 'test': 0}
            
        self.logger.info(f"🔍 Ditemukan {len(valid_files)} file valid dengan pasangan gambar/label")
        
        # Kelompokkan berdasarkan kelas jika stratify_by_class
        files_by_class = {}
        if stratify_by_class:
            for img_path, label_path in valid_files:
                classes = self._get_classes_from_label(label_path)
                main_class = classes[0] if classes else 'unknown'
                if main_class not in files_by_class:
                    files_by_class[main_class] = []
                files_by_class[main_class].append((img_path, label_path))
                    
            # Log distribusi kelas
            self.logger.info(f"📊 Distribusi kelas:")
            for cls, files in files_by_class.items():
                self.logger.info(f"   • Kelas {cls}: {len(files)} sampel")
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
        """Salin file ke direktori split."""
        self.logger.info(f"📂 Memindahkan {len(files)} file ke {split}...")
        
        with tqdm(files, desc=f"Copying to {split}") as pbar:
            for img_path, label_path in pbar:
                target_img = self.output_dir / split / 'images' / img_path.name
                target_label = self.output_dir / split / 'labels' / label_path.name
                
                try:
                    if use_symlinks:
                        if not target_img.exists(): target_img.symlink_to(img_path.resolve())
                        if not target_label.exists(): target_label.symlink_to(label_path.resolve())
                    else:
                        if not target_img.exists(): shutil.copy2(img_path, target_img)
                        if not target_label.exists(): shutil.copy2(label_path, target_label)
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memindahkan {img_path.name}: {str(e)}")
    
    def _get_classes_from_label(self, label_path: Path) -> List[int]:
        """Ekstrak ID kelas dari file label."""
        classes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes.append(int(float(parts[0])))
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
        return classes
    
    def _count_existing_splits(self) -> Dict[str, int]:
        """Hitung jumlah file dalam split yang sudah ada."""
        counts = {}
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.data_dir / split / 'images'
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