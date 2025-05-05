"""
File: smartcash/dataset/components/datasets/base_dataset.py
Deskripsi: Kelas dasar untuk semua dataset yang digunakan dalam proyek SmartCash
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from torch.utils.data import Dataset
from smartcash.common.logger import get_logger


class BaseDataset(Dataset):
    """Kelas dasar untuk semua dataset."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        logger = None,
        config: Optional[Dict] = None
    ):
        """
        Inisialisasi BaseDataset.
        
        Args:
            data_path: Path ke direktori data
            img_size: Ukuran gambar target
            mode: Mode dataset ('train', 'valid', 'test')
            transform: Transformasi yang akan diterapkan
            logger: Logger kustom (opsional)
            config: Konfigurasi aplikasi (opsional)
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Setup directories
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        # Periksa direktori
        self._check_directories()
        
        # Load data samples
        self.samples = self._load_samples()
        
        self.logger.info(f"✅ Dataset '{mode}' siap dengan {len(self.samples)} sampel valid")
    
    def __len__(self) -> int:
        """Mendapatkan jumlah sampel dalam dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Mengambil item dataset berdasarkan indeks.
        
        Args:
            idx: Indeks item yang akan diambil
            
        Returns:
            Dictionary berisi data sampel
        """
        raise NotImplementedError("Metode __getitem__ harus diimplementasikan di subclass")
    
    def _check_directories(self) -> None:
        """Periksa apakah direktori dataset valid."""
        if not self.images_dir.exists() or not self.labels_dir.exists():
            self.logger.warning(
                f"⚠️ Direktori data tidak lengkap:\n"
                f"   • Image dir: {self.images_dir} {'✅' if self.images_dir.exists() else '❌'}\n"
                f"   • Label dir: {self.labels_dir} {'✅' if self.labels_dir.exists() else '❌'}"
            )
    
    def _load_samples(self) -> List[Dict]:
        """
        Load sampel dataset. Metode ini harus diimplementasikan di subclass.
        
        Returns:
            List sampel valid dengan path dan metadata
        """
        raise NotImplementedError("Metode _load_samples harus diimplementasikan di subclass")
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load gambar dari file.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Array NumPy berisi gambar
        """
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize jika ukuran tidak sesuai dan tidak ada transform
        if not self.transform and (img.shape[0] != self.img_size[1] or img.shape[1] != self.img_size[0]):
            img = cv2.resize(img, self.img_size)
            
        return img
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Dapatkan statistik dataset.
        
        Returns:
            Dictionary berisi statistik dataset
        """
        return {
            'samples': len(self.samples),
            'mode': self.mode,
            'img_size': self.img_size,
            'data_path': str(self.data_path)
        }