"""
File: smartcash/dataset/services/explorer/base_explorer.py
Deskripsi: Kelas dasar untuk semua explorer dataset
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class BaseExplorer:
    """Kelas dasar untuk eksplorasi dataset."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi BaseExplorer.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger(self.__class__.__name__)
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        self.layer_config = get_layer_config()
    
    def _validate_directories(self, split: str) -> tuple:
        """
        Validasi direktori dataset dan kembalikan path serta status.
        
        Args:
            split: Split dataset
            
        Returns:
            Tuple (split_path, images_dir, labels_dir, valid)
        """
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        valid = images_dir.exists() and labels_dir.exists()
        if not valid:
            self.logger.warning(f"⚠️ Direktori tidak lengkap: {split_path}")
        
        return split_path, images_dir, labels_dir, valid
    
    def _get_valid_files(self, images_dir: Path, labels_dir: Path, sample_size: int = 0) -> List[Path]:
        """
        Dapatkan daftar file gambar valid dengan label.
        
        Args:
            images_dir: Direktori gambar
            labels_dir: Direktori label
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            List file gambar valid
        """
        # Cari file gambar
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return []
        
        # Ambil sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        return image_files