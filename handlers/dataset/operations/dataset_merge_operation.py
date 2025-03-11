# File: smartcash/handlers/dataset/operations/dataset_merge_operation.py
# Author: Alfrida Sabar
# Deskripsi: Operasi untuk menggabungkan dataset dari beberapa sumber atau split

import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger


class DatasetMergeOperation:
    """Operasi untuk menggabungkan dataset dari berbagai sumber atau split."""
    
    def __init__(self, data_dir: str, output_dir: Optional[str] = None, logger: Optional[SmartCashLogger] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'merged'
        self.logger = logger or SmartCashLogger(__name__)
    
    def merge_splits(self, splits: List[str] = ['train', 'valid', 'test'], unique_files: bool = True) -> Dict[str, int]:
        """Gabungkan beberapa split menjadi satu dataset flat."""
        # Buat direktori output
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        self.logger.info(f"🔄 Menggabungkan splits {splits} ke: {self.output_dir}")
        
        # Set untuk melacak file yang sudah ditambahkan (untuk unique_files)
        processed_files: Set[str] = set()
        all_files = []
        
        # Kumpulkan semua file dari setiap split
        for split in splits:
            images_dir, labels_dir = self.data_dir / split / 'images', self.data_dir / split / 'labels'
            if not (images_dir.exists() and labels_dir.exists()):
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap, melewati...")
                continue
                
            # Cari semua file gambar dengan label yang valid
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in images_dir.glob(ext):
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists() and (not unique_files or img_path.name not in processed_files):
                        all_files.append((img_path, label_path))
                        if unique_files:
                            processed_files.add(img_path.name)
        
        # Copy semua file ke direktori gabungan
        with tqdm(all_files, desc="Merging datasets") as pbar:
            for img_path, label_path in pbar:
                target_img, target_label = self.output_dir / 'images' / img_path.name, self.output_dir / 'labels' / label_path.name
                if not target_img.exists(): shutil.copy2(img_path, target_img)
                if not target_label.exists(): shutil.copy2(label_path, target_label)
        
        # Hitung jumlah file
        image_count = sum(1 for _ in (self.output_dir / 'images').glob('*.jpg')) + \
                      sum(1 for _ in (self.output_dir / 'images').glob('*.jpeg')) + \
                      sum(1 for _ in (self.output_dir / 'images').glob('*.png'))
        label_count = sum(1 for _ in (self.output_dir / 'labels').glob('*.txt'))
        
        self.logger.success(
            f"✅ Penggabungan dataset selesai:\n"
            f"   • Gambar: {image_count}\n"
            f"   • Label: {label_count}\n"
            f"   • Direktori: {self.output_dir}"
        )
        return {'images': image_count, 'labels': label_count, 'directory': str(self.output_dir)}
    
    def merge_datasets(self, dataset_dirs: List[str], unique_files: bool = True) -> Dict[str, int]:
        """Gabungkan beberapa dataset terpisah menjadi satu dataset."""
        # Buat direktori output
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        self.logger.info(f"🔄 Menggabungkan {len(dataset_dirs)} dataset ke: {self.output_dir}")
        
        processed_files, all_files = set(), []
        
        # Kumpulkan semua file dari setiap dataset
        for dataset_dir in dataset_dirs:
            dataset_path = Path(dataset_dir)
            
            # Struktur flat: images/ dan labels/ di root
            if (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
                self._collect_files_from_flat(dataset_path, all_files, processed_files, unique_files)
            # Struktur split: train/, valid/, test/ dengan images/ dan labels/ di dalamnya
            elif any((dataset_path / split).exists() for split in ['train', 'valid', 'test']):
                for split in ['train', 'valid', 'test']:
                    if (dataset_path / split).exists():
                        self._collect_files_from_flat(dataset_path / split, all_files, processed_files, unique_files)
            else:
                self.logger.warning(f"⚠️ Struktur dataset tidak dikenal di {dataset_dir}, melewati...")
        
        # Copy semua file ke direktori gabungan
        with tqdm(all_files, desc="Merging datasets") as pbar:
            for img_path, label_path in pbar:
                target_img, target_label = self.output_dir / 'images' / img_path.name, self.output_dir / 'labels' / label_path.name
                if not target_img.exists(): shutil.copy2(img_path, target_img)
                if not target_label.exists(): shutil.copy2(label_path, target_label)
        
        # Hitung jumlah file
        image_count = sum(1 for _ in (self.output_dir / 'images').glob('*.jpg')) + \
                      sum(1 for _ in (self.output_dir / 'images').glob('*.jpeg')) + \
                      sum(1 for _ in (self.output_dir / 'images').glob('*.png'))
        label_count = sum(1 for _ in (self.output_dir / 'labels').glob('*.txt'))
        
        self.logger.success(
            f"✅ Penggabungan dataset selesai:\n"
            f"   • Gambar: {image_count}\n"
            f"   • Label: {label_count}\n"
            f"   • Direktori: {self.output_dir}"
        )
        return {'images': image_count, 'labels': label_count, 'directory': str(self.output_dir)}
    
    def _collect_files_from_flat(self, dataset_path: Path, all_files: List, processed_files: Set[str], unique_files: bool):
        """Kumpulkan file dari dataset dengan struktur flat."""
        images_dir, labels_dir = dataset_path / 'images', dataset_path / 'labels'
        if not (images_dir.exists() and labels_dir.exists()): return
            
        # Cari semua file gambar dengan label yang valid
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in images_dir.glob(ext):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists() and (not unique_files or img_path.name not in processed_files):
                    all_files.append((img_path, label_path))
                    if unique_files:
                        processed_files.add(img_path.name)