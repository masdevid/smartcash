# File: smartcash/handlers/dataset/operations/dataset_merge_operation.py
# Author: Alfrida Sabar
# Deskripsi: Operasi untuk menggabungkan dataset dari beberapa sumber atau split

import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger


class DatasetMergeOperation:
    """
    Operasi untuk menggabungkan dataset dari berbagai sumber atau split.
    Mendukung penggabungan split yang terpisah menjadi satu dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetMergeOperation.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (jika None, gunakan data_dir/merged)
            logger: Logger kustom (opsional)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'merged'
        self.logger = logger or SmartCashLogger(__name__)
        
        self.logger.info(f"🔧 DatasetMergeOperation diinisialisasi: {self.data_dir}")
    
    def merge_splits(
        self, 
        splits: List[str] = ['train', 'valid', 'test'],
        unique_files: bool = True
    ) -> Dict[str, int]:
        """
        Gabungkan beberapa split menjadi satu dataset flat.
        
        Args:
            splits: List nama split yang akan digabungkan
            unique_files: Jika True, hindari file duplikat berdasarkan nama file
            
        Returns:
            Dict berisi jumlah file di direktori gabungan
        """
        # Buat direktori output
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔄 Menggabungkan splits {splits} ke: {self.output_dir}")
        
        # Set untuk melacak file yang sudah ditambahkan (untuk unique_files)
        processed_files: Set[str] = set()
        
        # Kumpulkan semua file dari setiap split
        all_files = []
        
        for split in splits:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap, melewati...")
                continue
                
            # Cari semua file gambar
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in images_dir.glob(ext):
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        # Cek apakah file sudah diproses jika unique_files=True
                        if unique_files and img_path.name in processed_files:
                            continue
                            
                        all_files.append((img_path, label_path))
                        if unique_files:
                            processed_files.add(img_path.name)
        
        # Copy semua file ke direktori gabungan
        with tqdm(all_files, desc="Merging datasets") as pbar:
            for img_path, label_path in pbar:
                target_img = self.output_dir / 'images' / img_path.name
                target_label = self.output_dir / 'labels' / label_path.name
                
                # Copy file jika belum ada
                if not target_img.exists():
                    shutil.copy2(img_path, target_img)
                if not target_label.exists():
                    shutil.copy2(label_path, target_label)
        
        # Hitung jumlah file
        image_count = len(list((self.output_dir / 'images').glob('*.jpg'))) + \
                      len(list((self.output_dir / 'images').glob('*.jpeg'))) + \
                      len(list((self.output_dir / 'images').glob('*.png')))
                      
        label_count = len(list((self.output_dir / 'labels').glob('*.txt')))
        
        self.logger.success(
            f"✅ Penggabungan dataset selesai:\n"
            f"   • Gambar: {image_count}\n"
            f"   • Label: {label_count}\n"
            f"   • Direktori: {self.output_dir}"
        )
        
        return {
            'images': image_count,
            'labels': label_count,
            'directory': str(self.output_dir)
        }
    
    def merge_datasets(
        self,
        dataset_dirs: List[str],
        unique_files: bool = True
    ) -> Dict[str, int]:
        """
        Gabungkan beberapa dataset terpisah menjadi satu dataset.
        
        Args:
            dataset_dirs: List direktori dataset yang akan digabungkan
            unique_files: Jika True, hindari file duplikat berdasarkan nama file
            
        Returns:
            Dict berisi jumlah file di direktori gabungan
        """
        # Buat direktori output
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔄 Menggabungkan {len(dataset_dirs)} dataset ke: {self.output_dir}")
        
        # Set untuk melacak file yang sudah ditambahkan (untuk unique_files)
        processed_files: Set[str] = set()
        
        # Kumpulkan semua file dari setiap dataset
        all_files = []
        
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
                target_img = self.output_dir / 'images' / img_path.name
                target_label = self.output_dir / 'labels' / label_path.name
                
                # Copy file jika belum ada
                if not target_img.exists():
                    shutil.copy2(img_path, target_img)
                if not target_label.exists():
                    shutil.copy2(label_path, target_label)
        
        # Hitung jumlah file
        image_count = len(list((self.output_dir / 'images').glob('*.jpg'))) + \
                      len(list((self.output_dir / 'images').glob('*.jpeg'))) + \
                      len(list((self.output_dir / 'images').glob('*.png')))
                      
        label_count = len(list((self.output_dir / 'labels').glob('*.txt')))
        
        self.logger.success(
            f"✅ Penggabungan dataset selesai:\n"
            f"   • Gambar: {image_count}\n"
            f"   • Label: {label_count}\n"
            f"   • Direktori: {self.output_dir}"
        )
        
        return {
            'images': image_count,
            'labels': label_count,
            'directory': str(self.output_dir)
        }
    
    def _collect_files_from_flat(
        self, 
        dataset_path: Path, 
        all_files: List, 
        processed_files: Set[str],
        unique_files: bool
    ) -> None:
        """
        Kumpulkan file dari dataset dengan struktur flat.
        
        Args:
            dataset_path: Path ke dataset
            all_files: List untuk menambahkan file yang ditemukan
            processed_files: Set file yang sudah diproses
            unique_files: Jika True, hindari file duplikat
        """
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            return
            
        # Cari semua file gambar
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in images_dir.glob(ext):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    # Cek apakah file sudah diproses jika unique_files=True
                    if unique_files and img_path.name in processed_files:
                        continue
                        
                    all_files.append((img_path, label_path))
                    if unique_files:
                        processed_files.add(img_path.name)