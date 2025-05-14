"""
File: smartcash/model/services/experiment/data_manager.py
Deskripsi: Komponen untuk mengelola data eksperimen
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from smartcash.common.logger import get_logger


class ExperimentDataManager:
    """
    Komponen untuk mengelola data eksperimen.
    
    Bertanggung jawab untuk:
    - Loading dan preprocessing dataset
    - Pembuatan data loaders
    - Train-validation-test split
    - Augmentasi data
    """
    
    def __init__(
        self,
        dataset_path: str = "data",
        batch_size: int = 16,
        val_split: float = 0.2,
        test_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 4,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment data manager.
        
        Args:
            dataset_path: Path ke dataset
            batch_size: Ukuran batch default
            val_split: Rasio data validasi
            test_split: Rasio data test
            shuffle: Flag untuk mengacak data
            num_workers: Jumlah worker untuk DataLoader
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.logger = logger or get_logger("experiment_data_manager")
        
        # Dataset dan DataLoader
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.logger.debug(f"🔄 ExperimentDataManager diinisialisasi (dataset_path={dataset_path})")
    
    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        transform: Optional[Callable] = None
    ) -> Any:
        """
        Load dataset dari path.
        
        Args:
            dataset_path: Path ke dataset (opsional, default: self.dataset_path)
            transform: Transformasi yang akan diterapkan ke dataset
            
        Returns:
            Dataset yang dimuat
        """
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        self.logger.info(f"📂 Loading dataset dari {self.dataset_path}")
        
        # Implementasi untuk load dataset harus disesuaikan dengan jenis dataset
        # Berikut adalah contoh dummy implementation
        
        # Dummy dataset
        self.dataset = TensorDataset(
            torch.from_numpy(np.random.rand(200, 3, 640, 640).astype(np.float32)),
            torch.from_numpy(np.random.rand(200, 10, 5).astype(np.float32))
        )
        
        self.logger.info(f"✅ Dataset dimuat: {len(self.dataset)} sampel")
        
        return self.dataset
    
    def create_data_splits(
        self,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
        seed: int = 42
    ) -> Tuple[Any, Any, Any]:
        """
        Buat train-validation-test split.
        
        Args:
            val_split: Rasio data validasi (opsional)
            test_split: Rasio data test (opsional)
            seed: Random seed untuk reproducibility
            
        Returns:
            Tuple (train_dataset, val_dataset, test_dataset)
        """
        if not self.dataset:
            raise ValueError("Dataset belum dimuat. Panggil load_dataset terlebih dahulu.")
        
        # Gunakan parameter yang disediakan atau default
        val_split = val_split if val_split is not None else self.val_split
        test_split = test_split if test_split is not None else self.test_split
        
        # Hitung jumlah sampel untuk setiap split
        dataset_size = len(self.dataset)
        test_size = int(dataset_size * test_split)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - test_size - val_size
        
        # Buat split
        generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        self.logger.info(
            f"✅ Dataset split dibuat:\n"
            f"   • Train: {train_size} sampel\n"
            f"   • Validation: {val_size} sampel\n"
            f"   • Test: {test_size} sampel"
        )
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def create_data_loaders(
        self,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Buat data loaders untuk training, validation, dan testing.
        
        Args:
            batch_size: Ukuran batch (opsional)
            shuffle: Flag untuk mengacak data (opsional)
            num_workers: Jumlah worker (opsional)
            
        Returns:
            Tuple (train_loader, val_loader, test_loader)
        """
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            # Jika dataset belum di-split, coba load dan split
            if not self.dataset:
                self.load_dataset()
            self.create_data_splits()
        
        # Gunakan parameter yang disediakan atau default
        batch_size = batch_size if batch_size is not None else self.batch_size
        shuffle = shuffle if shuffle is not None else self.shuffle
        num_workers = num_workers if num_workers is not None else self.num_workers
        
        # Buat data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        self.logger.info(
            f"✅ Data loaders dibuat:\n"
            f"   • Train: {len(self.train_loader)} batch\n"
            f"   • Validation: {len(self.val_loader)} batch\n"
            f"   • Test: {len(self.test_loader)} batch"
        )
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_data_loaders(
        self,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load dataset, buat splits, dan kembalikan data loaders dalam satu langkah.
        
        Args:
            dataset_path: Path ke dataset (opsional)
            batch_size: Ukuran batch (opsional)
            
        Returns:
            Tuple (train_loader, val_loader, test_loader)
        """
        # Load dataset jika belum
        if dataset_path or not self.dataset:
            self.load_dataset(dataset_path)
        
        # Buat splits jika belum
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            self.create_data_splits()
        
        # Buat data loaders
        return self.create_data_loaders(batch_size)
    
    def get_class_names(self) -> List[str]:
        """
        Dapatkan daftar nama kelas dari dataset.
        
        Returns:
            List nama kelas
        """
        # Implementasi ini bergantung pada jenis dataset
        # Contoh dummy implementation
        return [f'class_{i}' for i in range(10)]
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik dataset.
        
        Returns:
            Dictionary statistik dataset
        """
        if not self.dataset:
            raise ValueError("Dataset belum dimuat")
            
        stats = {
            "total_samples": len(self.dataset),
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
            "class_distribution": self._get_class_distribution(),
            "input_shape": self._get_input_shape()
        }
        
        return stats
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """
        Hitung distribusi kelas dalam dataset.
        
        Returns:
            Dictionary distribusi kelas
        """
        # Implementasi ini bergantung pada jenis dataset
        # Contoh dummy implementation
        return {f'class_{i}': 20 for i in range(10)}
    
    def _get_input_shape(self) -> List[int]:
        """
        Dapatkan bentuk input dari dataset.
        
        Returns:
            List bentuk input
        """
        # Implementasi ini bergantung pada jenis dataset
        # Contoh dummy implementation
        return [3, 640, 640]