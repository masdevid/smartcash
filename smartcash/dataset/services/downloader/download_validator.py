"""
File: smartcash/dataset/services/downloader/download_validator.py
Deskripsi: Komponen untuk memvalidasi integritas dan struktur dataset yang didownload
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS


class DownloadValidator:
    """
    Validator untuk memeriksa integritas dan struktur dataset yang didownload.
    Memastikan dataset lengkap dan sesuai dengan format yang diharapkan.
    """
    
    def __init__(self, logger=None):
        """
        Inisialisasi DownloadValidator.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("download_validator")
    
    def verify_download(self, download_dir: Union[str, Path], metadata: Dict) -> bool:
        """
        Verifikasi integritas hasil download berdasarkan metadata.
        
        Args:
            download_dir: Direktori hasil download
            metadata: Metadata dataset dari Roboflow
            
        Returns:
            Boolean yang menunjukkan keberhasilan verifikasi
        """
        download_path = Path(download_dir)
        
        if not download_path.exists():
            self.logger.error(f"❌ Direktori download tidak ditemukan: {download_path}")
            return False
        
        try:
            # Cek struktur direktori
            expected_dirs = DEFAULT_SPLITS
            for split in expected_dirs:
                split_dir = download_path / split
                
                if not split_dir.exists():
                    self.logger.warning(f"⚠️ Direktori {split} tidak ditemukan")
                    continue
                
                # Cek subdirektori images dan labels
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                if not images_dir.exists():
                    self.logger.warning(f"⚠️ Direktori {split}/images tidak ditemukan")
                    continue
                    
                if not labels_dir.exists():
                    self.logger.warning(f"⚠️ Direktori {split}/labels tidak ditemukan")
                    continue
                
                # Cek jumlah file
                img_count = len(list(images_dir.glob('*.*')))
                label_count = len(list(labels_dir.glob('*.txt')))
                
                # Verifikasi dengan metadata jika tersedia
                expected_count = metadata.get('version', {}).get('splits', {}).get(split)
                
                if expected_count is not None:
                    if img_count < expected_count:
                        self.logger.warning(
                            f"⚠️ Jumlah gambar di {split} ({img_count}) "
                            f"kurang dari yang diharapkan ({expected_count})"
                        )
                
                # Cek kecocokan gambar dan label
                orphan_images = 0
                orphan_labels = 0
                
                # Sampling jika jumlah file terlalu banyak
                max_sample = 100
                
                if img_count > max_sample:
                    # Sampling gambar
                    image_files = list(images_dir.glob('*.*'))[:max_sample]
                    
                    for img_file in image_files:
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if not label_file.exists():
                            orphan_images += 1
                else:
                    # Cek semua gambar
                    for img_file in images_dir.glob('*.*'):
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if not label_file.exists():
                            orphan_images += 1
                
                # Cek label yang tidak memiliki gambar
                if label_count > max_sample:
                    # Sampling label
                    label_files = list(labels_dir.glob('*.txt'))[:max_sample]
                    
                    for label_file in label_files:
                        # Cek semua kemungkinan ekstensi gambar
                        found = False
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            img_file = images_dir / f"{label_file.stem}{ext}"
                            if img_file.exists():
                                found = True
                                break
                        
                        if not found:
                            orphan_labels += 1
                else:
                    # Cek semua label
                    for label_file in labels_dir.glob('*.txt'):
                        found = False
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            img_file = images_dir / f"{label_file.stem}{ext}"
                            if img_file.exists():
                                found = True
                                break
                        
                        if not found:
                            orphan_labels += 1
                
                # Log hasil
                if orphan_images > 0 or orphan_labels > 0:
                    self.logger.warning(
                        f"⚠️ Ditemukan {orphan_images} gambar tanpa label dan "
                        f"{orphan_labels} label tanpa gambar di {split}"
                    )
                
                # Log statistik
                self.logger.info(
                    f"📊 Statistik {split}: {img_count} gambar, {label_count} label, "
                    f"orphan images: {orphan_images}, orphan labels: {orphan_labels}"
                )
            
            # Cek minimal satu split harus ada dan valid
            valid_splits = []
            for split in expected_dirs:
                images_dir = download_path / split / 'images'
                labels_dir = download_path / split / 'labels'
                
                if (images_dir.exists() and labels_dir.exists() and 
                    len(list(images_dir.glob('*.*'))) > 0 and 
                    len(list(labels_dir.glob('*.txt'))) > 0):
                    valid_splits.append(split)
            
            if not valid_splits:
                self.logger.error("❌ Tidak ada split yang valid dalam dataset")
                return False
                
            self.logger.success(f"✅ Verifikasi download berhasil, split valid: {', '.join(valid_splits)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat verifikasi download: {str(e)}")
            return False
    
    def verify_local_dataset(self, data_dir: Union[str, Path]) -> bool:
        """
        Verifikasi struktur dataset lokal.
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Boolean yang menunjukkan hasil verifikasi
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            self.logger.error(f"❌ Direktori dataset tidak ditemukan: {data_path}")
            return False
        
        try:
            # Cek struktur direktori
            valid_splits = []
            
            for split in DEFAULT_SPLITS:
                split_dir = data_path / split
                
                if not split_dir.exists():
                    self.logger.warning(f"⚠️ Direktori {split} tidak ditemukan")
                    continue
                
                # Cek subdirektori images dan labels
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                if not images_dir.exists():
                    self.logger.warning(f"⚠️ Direktori {split}/images tidak ditemukan")
                    continue
                    
                if not labels_dir.exists():
                    self.logger.warning(f"⚠️ Direktori {split}/labels tidak ditemukan")
                    continue
                
                # Cek jumlah file
                img_count = len(list(images_dir.glob('*.*')))
                label_count = len(list(labels_dir.glob('*.txt')))
                
                if img_count == 0 or label_count == 0:
                    self.logger.warning(f"⚠️ Split {split} kosong: {img_count} gambar, {label_count} label")
                    continue
                
                # Log statistik
                self.logger.info(f"📊 Statistik {split}: {img_count} gambar, {label_count} label")
                valid_splits.append(split)
            
            # Cek minimal satu split harus ada dan valid
            if not valid_splits:
                self.logger.error("❌ Tidak ada split yang valid dalam dataset")
                return False
                
            self.logger.success(f"✅ Verifikasi dataset lokal berhasil, split valid: {', '.join(valid_splits)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat verifikasi dataset lokal: {str(e)}")
            return False
    
    def verify_dataset_structure(self, dataset_dir: Union[str, Path]) -> bool:
        """
        Verifikasi struktur direktori dataset.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Boolean yang menunjukkan hasil verifikasi
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            self.logger.error(f"❌ Direktori dataset tidak ditemukan: {dataset_path}")
            return False
        
        try:
            # Cek struktur direktori
            valid_structure = True
            
            # Cek apakah ada struktur train/valid/test
            has_splits = True
            for split in DEFAULT_SPLITS:
                if not (dataset_path / split).exists():
                    has_splits = False
                    break
            
            if has_splits:
                # Cek struktur dengan split
                for split in DEFAULT_SPLITS:
                    split_dir = dataset_path / split
                    
                    # Cek subdirektori images dan labels
                    images_dir = split_dir / 'images'
                    labels_dir = split_dir / 'labels'
                    
                    if not images_dir.exists() or not labels_dir.exists():
                        self.logger.warning(f"⚠️ Split {split} tidak lengkap")
                        valid_structure = False
            elif (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
                # Struktur tanpa split (hanya images dan labels di root)
                self.logger.warning(
                    f"⚠️ Dataset memiliki struktur tanpa split, "
                    f"sebaiknya gunakan struktur train/valid/test"
                )
                valid_structure = True
            else:
                # Struktur tidak dikenal
                self.logger.error(f"❌ Struktur dataset tidak dikenal")
                valid_structure = False
            
            return valid_structure
            
        except Exception as e:
            self.logger.error(f"❌ Error saat verifikasi struktur dataset: {str(e)}")
            return False
    
    def is_dataset_available(
        self,
        data_dir: Union[str, Path],
        verify_content: bool = False
    ) -> bool:
        """
        Cek apakah dataset tersedia di direktori.
        
        Args:
            data_dir: Direktori dataset
            verify_content: Apakah memverifikasi konten dataset
            
        Returns:
            Boolean yang menunjukkan ketersediaan dataset
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return False
        
        # Cek struktur dasar
        for split in DEFAULT_SPLITS:
            if not (data_path / split).exists():
                return False
                
            # Cek subdirektori
            if not (data_path / split / 'images').exists() or not (data_path / split / 'labels').exists():
                return False
        
        # Jika tidak perlu verifikasi konten lebih lanjut
        if not verify_content:
            return True
            
        # Verifikasi konten (minimal ada file di train)
        train_images = list((data_path / 'train' / 'images').glob('*.*'))
        train_labels = list((data_path / 'train' / 'labels').glob('*.txt'))
        
        return len(train_images) > 0 and len(train_labels) > 0
    
    def get_dataset_stats(self, dataset_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Dapatkan statistik dataset.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Dictionary berisi statistik dataset
        """
        dataset_path = Path(dataset_dir)
        stats = {'total_images': 0, 'total_labels': 0, 'splits': {}}
        
        if not dataset_path.exists():
            return stats
        
        # Cek struktur dengan split
        for split in DEFAULT_SPLITS:
            split_dir = dataset_path / split
            
            if not split_dir.exists():
                continue
                
            # Cek subdirektori
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
                
            # Hitung file
            img_count = len(list(images_dir.glob('*.*')))
            label_count = len(list(labels_dir.glob('*.txt')))
            
            stats['splits'][split] = {'images': img_count, 'labels': label_count}
            stats['total_images'] += img_count
            stats['total_labels'] += label_count
        
        return stats
    
    def get_local_stats(self, data_dir: Union[str, Path]) -> Dict[str, int]:
        """
        Dapatkan statistik dataset lokal (jumlah sampel per split).
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Dictionary berisi jumlah sampel per split
        """
        data_path = Path(data_dir)
        stats = {}
        
        if not data_path.exists():
            return stats
        
        for split in DEFAULT_SPLITS:
            images_dir = data_path / split / 'images'
            
            if not images_dir.exists():
                stats[split] = 0
                continue
                
            # Hitung gambar
            img_count = len(list(images_dir.glob('*.*')))
            stats[split] = img_count
        
        return stats