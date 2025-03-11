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
        # Preprocess ekstensi gambar ke format suffix yang konsisten
        self.img_suffixes = {f".{ext.lower().split('*.')[-1].lstrip('.')}" for ext in IMG_EXTENSIONS}
    
    def verify_download(self, output_dir: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """
        Verifikasi hasil download dataset.
        
        Args:
            output_dir: Direktori output download
            metadata: Metadata dari Roboflow (untuk verifikasi jumlah file)
            
        Returns:
            Boolean yang menunjukkan apakah download valid
        """
        output_path = Path(output_dir)  # Convert to Path object
        
        # Periksa struktur dasar
        if not self._optimized_check(output_path): return False
        
        # Hitung file per split untuk validasi metadata
        split_counts = {
            split: sum(1 for f in (output_path / split / 'images').iterdir() 
                       if f.is_file() and f.suffix.lower() in self.img_suffixes)
            for split in DEFAULT_SPLITS
        }
        
        # Log jumlah file
        if self.logger:
            [self.logger.info(f"📊 Split {s}: {c} gambar") for s, c in split_counts.items()]
        
        # Verifikasi dengan metadata jika tersedia
        if metadata and 'version' in metadata and 'splits' in metadata['version']:
            return all(
                abs(c - metadata['version']['splits'][s]) <= metadata['version']['splits'][s] * 0.1
                for s, c in split_counts.items()
                if s in metadata['version']['splits'] and metadata['version']['splits'][s] > 0
            )
        return True
    
    def _optimized_check(self, output_path: Path) -> bool:
        """
        Periksa struktur dataset dengan efisien.
        
        Args:
            output_path: Path ke direktori dataset
            
        Returns:
            Boolean yang menunjukkan validitas struktur
        """
        return all(
            (output_path / split).is_dir() and
            (output_path / split / 'images').is_dir() and
            (output_path / split / 'labels').is_dir() and
            any(f.is_file() and f.suffix.lower() in self.img_suffixes 
                for f in (output_path / split / 'images').iterdir()) and
            any(f.is_file() and f.suffix.lower() == '.txt' 
                for f in (output_path / split / 'labels').iterdir())
            for split in DEFAULT_SPLITS
        )
    
    def is_dataset_available(self, data_dir: Union[str, Path], verify_content: bool = False) -> bool:
        """
        Periksa apakah dataset tersedia di direktori lokal.
        
        Args:
            data_dir: Direktori dataset
            verify_content: Flag untuk memeriksa konten secara lebih detail
            
        Returns:
            Boolean yang menunjukkan apakah dataset tersedia
        """
        return self._optimized_check(Path(data_dir))  # Convert to Path object
    
    def get_local_stats(self, data_dir: Union[str, Path]) -> Dict[str, int]:
        """
        Dapatkan statistik dasar dataset lokal.
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Dict dengan jumlah gambar per split
        """
        data_path = Path(data_dir)  # Convert to Path object
        stats = {
            s: sum(1 for f in (data_path / s / 'images').iterdir() 
                   if f.is_file() and f.suffix.lower() in self.img_suffixes)
            for s in DEFAULT_SPLITS
        }
        if self.logger:
            self.logger.info(f"📊 Dataset: train={stats['train']}, valid={stats['valid']}, test={stats['test']} gambar")
        return stats
    
    def verify_local_dataset(self, data_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Verifikasi dataset lokal dan berikan statistik dasar.
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Dict statistik dataset
        """
        data_path = Path(data_dir)  # Convert to Path object
        valid = self._optimized_check(data_path)
        stats = self.get_local_stats(data_path)
        invalid_splits = [s for s, c in stats.items() if c == 0]
        return {
            'valid': valid and not invalid_splits,
            'splits': {s: {'images': c, 'valid': c > 0} for s, c in stats.items()},
            'message': 'Dataset valid' if valid and not invalid_splits else
                       f"Invalid splits: {', '.join(invalid_splits)}" if invalid_splits else
                       'Struktur dataset tidak valid'
        }
    
    def get_dataset_stats(self, dataset_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Dapatkan statistik umum dataset.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Dict berisi statistik dataset
        """
        stats = self.get_local_stats(dataset_dir)
        return {
            'total_images': sum(stats.values()),
            'splits': stats,
            'structure_valid': self._optimized_check(Path(dataset_dir))  # Convert to Path object
        }