"""
File: smartcash/handlers/dataset/core/download_core/download_validator.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk validasi hasil download dataset
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any

from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS, IMG_EXTENSIONS


class DownloadValidator:
    """Komponen untuk validasi hasil download dataset."""
    
    def __init__(self, logger=None):
        """Inisialisasi DownloadValidator."""
        self.logger = logger
        self.allowed_img_suffixes = self._process_img_extensions(IMG_EXTENSIONS)
    
    def _process_img_extensions(self, extensions):
        """Process IMG_EXTENSIONS into a set of lowercase suffixes with leading dots."""
        allowed = set()
        for ext in extensions:
            # Split the extension to handle patterns like '*.jpg'
            parts = ext.lower().split('*.')
            if len(parts) > 1:
                suffix = f".{parts[-1].lstrip('.')}"
            else:
                suffix = f".{parts[0].lstrip('.')}"
            allowed.add(suffix)
        return allowed
    
    def verify_download(self, output_dir: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """
        Verifikasi hasil download dataset.
        
        Args:
            output_dir: Direktori output download
            metadata: Metadata dari Roboflow (untuk verifikasi jumlah file)
            
        Returns:
            Boolean yang menunjukkan apakah download valid
        """
        output_path = Path(output_dir)
        
        # Periksa struktur dasar
        if not self._optimized_check(output_path, self.allowed_img_suffixes):
            return False
        
        # Hitung file per split untuk validasi metadata
        split_counts = {}
        for split in DEFAULT_SPLITS:
            img_dir = output_path / split / 'images'
            
            # Hitung gambar yang valid
            try:
                img_count = sum(1 for file in img_dir.iterdir() 
                               if file.is_file() and file.suffix.lower() in self.allowed_img_suffixes)
            except (NotADirectoryError, PermissionError):
                img_count = 0
            
            split_counts[split] = img_count
            
            # Log jumlah file
            if self.logger:
                self.logger.info(f"📊 Split {split}: {img_count} gambar")
        
        # Verifikasi dengan metadata jika tersedia
        if metadata and 'version' in metadata and 'splits' in metadata['version']:
            expected_counts = metadata['version']['splits']
            for split, expected in expected_counts.items():
                if split in split_counts:
                    actual = split_counts[split]
                    # Toleransi 10% perbedaan jumlah
                    if expected > 0 and abs(actual - expected) > expected * 0.1:
                        if self.logger:
                            self.logger.warning(f"⚠️ Jumlah gambar {split} tidak sesuai: {actual} vs {expected} (expected)")
                        return False
        
        return True
    
    def _optimized_check(self, output_path: Path, img_suffixes: set) -> bool:
        """
        Periksa struktur dataset dengan efisien.
        
        Args:
            output_path: Path ke direktori dataset
            img_suffixes: Set ekstensi gambar yang didukung (dengan leading dot)
            
        Returns:
            Boolean yang menunjukkan validitas struktur
        """
        for split in DEFAULT_SPLITS:
            split_path = output_path / split
            img_dir = split_path / 'images'
            lbl_dir = split_path / 'labels'
            
            # Cek direktori
            if not (split_path.is_dir() and img_dir.is_dir() and lbl_dir.is_dir()):
                if self.logger:
                    self.logger.warning(f"⚠️ Struktur direktori tidak lengkap: {split}")
                return False
                
            # Cek file gambar
            try:
                has_images = any(
                    file.is_file() and file.suffix.lower() in img_suffixes
                    for file in img_dir.iterdir()
                )
            except (NotADirectoryError, PermissionError):
                if self.logger:
                    self.logger.warning(f"⚠️ Direktori gambar tidak dapat diakses: {img_dir}")
                return False
                
            # Cek file label
            try:
                has_labels = any(
                    file.is_file() and file.suffix.lower() == '.txt'
                    for file in lbl_dir.iterdir()
                )
            except (NotADirectoryError, PermissionError):
                if self.logger:
                    self.logger.warning(f"⚠️ Direktori label tidak dapat diakses: {lbl_dir}")
                return False
            
            if not (has_images and has_labels):
                if self.logger:
                    self.logger.warning(f"⚠️ Tidak ada file gambar/label di {split}")
                return False
                
        return True
    
    def is_dataset_available(self, data_dir: Union[str, Path], verify_content: bool = False) -> bool:
        """
        Periksa apakah dataset tersedia di direktori lokal.
        
        Args:
            data_dir: Direktori dataset
            verify_content: Flag untuk memeriksa konten secara lebih detail
            
        Returns:
            Boolean yang menunjukkan apakah dataset tersedia
        """
        data_path = Path(data_dir)
        return self._optimized_check(data_path, self.allowed_img_suffixes)
    
    def get_local_stats(self, data_dir: Union[str, Path]) -> Dict[str, int]:
        """
        Dapatkan statistik dasar dataset lokal.
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Dict dengan jumlah gambar per split
        """
        data_path = Path(data_dir)
        stats = {}
        
        for split in DEFAULT_SPLITS:
            img_dir = data_path / split / 'images'
            stats[split] = 0
            
            if not img_dir.exists() or not img_dir.is_dir():
                continue
                
            try:
                count = sum(1 for file in img_dir.iterdir() 
                           if file.is_file() and file.suffix.lower() in self.allowed_img_suffixes)
                stats[split] = count
            except (PermissionError, NotADirectoryError):
                continue
        
        if self.logger:
            self.logger.info(
                f"📊 Dataset: train={stats.get('train', 0)}, "
                f"valid={stats.get('valid', 0)}, "
                f"test={stats.get('test', 0)} gambar"
            )
            
        return stats
    
    def verify_local_dataset(self, data_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Verifikasi dataset lokal dan berikan statistik dasar.
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Dict statistik dataset
        """
        data_path = Path(data_dir)
        stats = {
            'valid': True,
            'splits': {},
            'message': 'Dataset valid'
        }
        
        if not self._optimized_check(data_path, self.allowed_img_suffixes):
            stats.update({'valid': False, 'message': 'Struktur dataset tidak valid'})
            return stats
        
        split_stats = self.get_local_stats(data_dir)
        
        for split in DEFAULT_SPLITS:
            count = split_stats.get(split, 0)
            stats['splits'][split] = {
                'images': count,
                'valid': count > 0
            }
            if count == 0:
                stats.update({'valid': False, 'message': f'Split {split} tidak memiliki gambar'})
        
        return stats
    
    def get_dataset_stats(self, dataset_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Dapatkan statistik umum dataset.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Dict berisi statistik dataset
        """
        local_stats = self.get_local_stats(dataset_dir)
        return {
            'total_images': sum(local_stats.values()),
            'splits': local_stats,
            'structure_valid': self._optimized_check(Path(dataset_dir), self.allowed_img_suffixes)
        }