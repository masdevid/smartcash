# File: smartcash/handlers/dataset/explorers/validation_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer khusus untuk validasi integritas dataset

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class ValidationExplorer(BaseExplorer):
    """Explorer khusus untuk validasi integritas dataset."""
    
    def explore(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Melakukan validasi integritas dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel untuk validasi (0 = semua)
            
        Returns:
            Dict hasil validasi
        """
        self.logger.info(f"🔍 Validasi integritas dataset: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar di split {split}")
            return {'error': f"Tidak ada file gambar di split {split}"}
        
        # Batasi sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk validasi")
        
        # Validasi integritas dataset
        validation_stats = self._analyze_validation_stats(images_dir, labels_dir, image_files)
        
        # Log hasil validasi
        valid_labels = validation_stats.get('valid_labels', 0)
        invalid_labels = validation_stats.get('invalid_labels', 0)
        missing_labels = validation_stats.get('missing_labels', 0)
        corrupt_images = validation_stats.get('corrupt_images', 0)
        total_images = validation_stats.get('total_images', 0)
        
        self.logger.info(
            f"📊 Hasil validasi dataset '{split}':\n"
            f"   • Total gambar: {total_images}\n"
            f"   • Label valid: {valid_labels} ({valid_labels/max(1, total_images)*100:.1f}%)\n"
            f"   • Label tidak valid: {invalid_labels} ({invalid_labels/max(1, total_images)*100:.1f}%)\n"
            f"   • Label hilang: {missing_labels} ({missing_labels/max(1, total_images)*100:.1f}%)\n"
            f"   • Gambar rusak: {corrupt_images} ({corrupt_images/max(1, total_images)*100:.1f}%)"
        )
        
        return validation_stats
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik dasar untuk semua split dataset.
        
        Returns:
            Dict statistik per split
        """
        self.logger.info("📊 Mengumpulkan statistik semua split dataset...")
        
        splits = ['train', 'valid', 'test']
        stats = {}
        
        for split in splits:
            split_dir = self._get_split_path(split)
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                stats[split] = {
                    'images': 0,
                    'labels': 0,
                    'status': 'missing'
                }
                continue
            
            # Hitung file
            image_count = sum(1 for _ in images_dir.glob('*.jpg')) + \
                          sum(1 for _ in images_dir.glob('*.jpeg')) + \
                          sum(1 for _ in images_dir.glob('*.png'))
                          
            label_count = sum(1 for _ in labels_dir.glob('*.txt'))
            
            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'status': 'valid' if image_count > 0 and label_count > 0 else 'empty'
            }
        
        self.logger.info(
            f"📊 Ringkasan statistik dataset:\n"
            f"   • Train: {stats.get('train', {}).get('images', 0)} gambar, {stats.get('train', {}).get('labels', 0)} label\n"
            f"   • Valid: {stats.get('valid', {}).get('images', 0)} gambar, {stats.get('valid', {}).get('labels', 0)} label\n"
            f"   • Test: {stats.get('test', {}).get('images', 0)} gambar, {stats.get('test', {}).get('labels', 0)} label"
        )
        
        return stats
    
    def _analyze_validation_stats(self, images_dir: Path, labels_dir: Path, image_files: List[Path]) -> Dict[str, int]:
        """
        Analisis statistik validasi dasar.
        
        Args:
            images_dir: Direktori gambar
            labels_dir: Direktori label
            image_files: List file gambar
            
        Returns:
            Dict berisi statistik validasi
        """
        # Hitung total gambar
        total_images = len(image_files)
        
        # Hitung label yang valid
        valid_labels = 0
        invalid_labels = 0
        missing_labels = 0
        corrupt_images = 0
        
        for img_path in tqdm(image_files, desc="Validasi dataset"):
            # Cek label yang sesuai
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                missing_labels += 1
                continue
            
            # Validasi file label
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                if lines:  # File tidak kosong
                    # Cek format setiap baris
                    all_valid = True
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:  # Format YOLO: class_id x y w h
                            all_valid = False
                            break
                    
                    if all_valid:
                        valid_labels += 1
                    else:
                        invalid_labels += 1
                else:
                    # File kosong
                    invalid_labels += 1
            except Exception:
                invalid_labels += 1
            
            # Coba membuka gambar untuk memastikan tidak corrupt
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_images += 1
            except Exception:
                corrupt_images += 1
        
        return {
            'total_images': total_images,
            'valid_labels': valid_labels,
            'invalid_labels': invalid_labels,
            'missing_labels': missing_labels,
            'corrupt_images': corrupt_images
        }