"""
File: smartcash/dataset/components/samplers/balanced_sampler.py
Deskripsi: Implementasi sampling batch dengan balancing kelas untuk mengatasi ketidakseimbangan dataset
"""

import numpy as np
from typing import Dict, List, Optional, Any, Iterator, Callable
import torch
from torch.utils.data import Sampler, Dataset

from smartcash.common.logger import get_logger


class BalancedBatchSampler(Sampler):
    """
    Sampler yang menghasilkan batch yang seimbang dari segi kelas.
    Memastikan setiap batch memiliki jumlah sampel per kelas yang sama atau mendekati.
    """
    
    def __init__(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        class_getter: Callable[[int], int],
        drop_last: bool = False,
        logger = None
    ):
        """
        Inisialisasi BalancedBatchSampler.
        
        Args:
            dataset: Dataset untuk sampling
            batch_size: Ukuran batch
            class_getter: Fungsi untuk mendapatkan kelas dari indeks dataset
            drop_last: Apakah membuang batch terakhir jika tidak lengkap
            logger: Logger kustom (opsional)
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_getter = class_getter
        self.drop_last = drop_last
        self.logger = logger or get_logger()
        
        # Map kelas ke indeks sampel
        self.class_indices = self._build_class_indices()
        self.num_classes = len(self.class_indices)
        
        # Hitung sampel per kelas per batch
        self.target_per_class = max(1, batch_size // self.num_classes)
        self.actual_batch_size = self.target_per_class * self.num_classes

        # Penyesuaian untuk kelas dengan jumlah sampel yang kurang
        min_samples = min(len(indices) for indices in self.class_indices.values())
        if min_samples < self.target_per_class:
            self.logger.warning(
                f"⚠️ Minimal sampel per kelas ({min_samples}) kurang dari target per batch "
                f"({self.target_per_class}). Gunakan semua sampel kelas minoritas."
            )
            self.target_per_class = min_samples
            self.actual_batch_size = self.target_per_class * self.num_classes
        
        # Adaptasi ukuran batch jika tidak habis dibagi
        if batch_size != self.actual_batch_size:
            self.logger.info(
                f"ℹ️ Ukuran batch disesuaikan dari {batch_size} → {self.actual_batch_size} "
                f"({self.target_per_class} sampel × {self.num_classes} kelas)"
            )
        
        # Hitung jumlah batch
        self.num_batches = self._compute_num_batches()
        
        self.logger.info(
            f"⚖️ BalancedBatchSampler siap dengan {self.num_batches} batch seimbang, "
            f"{self.num_classes} kelas, {self.target_per_class} sampel/kelas/batch"
        )
    
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """
        Membangun mapping dari kelas ke indeks sampel.
        
        Returns:
            Dictionary berisi {class_id: [index1, index2, ...]}
        """
        class_indices = {}
        
        # Klasifikasikan setiap sampel berdasarkan kelasnya
        for idx in range(len(self.dataset)):
            try:
                class_id = self.class_getter(idx)
                if class_id not in class_indices:
                    class_indices[class_id] = []
                class_indices[class_id].append(idx)
            except Exception as e:
                self.logger.debug(f"⚠️ Error mendapatkan kelas untuk indeks {idx}: {str(e)}")
        
        # Log statistik kelas
        self.logger.info(f"📊 Distribusi kelas dalam dataset:")
        for cls, indices in sorted(class_indices.items(), key=lambda x: len(x[1]), reverse=True):
            samples = len(indices)
            self.logger.info(f"   • Kelas {cls}: {samples} sampel")
        
        return class_indices
    
    def _compute_num_batches(self) -> int:
        """
        Hitung jumlah batch dalam satu epoch.
        
        Returns:
            Jumlah batch
        """
        # Hitung batch per kelas (berapa batch yang bisa dibuat untuk setiap kelas)
        batches_per_class = [len(indices) // self.target_per_class for indices in self.class_indices.values()]
        min_batches = min(batches_per_class)
        
        # Jika drop_last = True, gunakan jumlah batch minimum
        # Jika drop_last = False, tambah 1 batch jika ada sisa
        if min_batches == 0:
            return 1  # Minimal 1 batch
        
        if not self.drop_last:
            # Periksa apakah ada sisa di kelas manapun
            has_remainder = any(len(indices) % self.target_per_class > 0 for indices in self.class_indices.values())
            return min_batches + (1 if has_remainder else 0)
        
        return min_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterasi batch yang seimbang.
        
        Returns:
            Iterator berisi list indeks sampel
        """
        # Shuffle indices per kelas
        indices_per_class = {
            cls: indices.copy() for cls, indices in self.class_indices.items()
        }
        
        for cls, indices in indices_per_class.items():
            np.random.shuffle(indices)
        
        # State untuk tracking berapa sampel yang sudah diambil dari tiap kelas
        pointers = {cls: 0 for cls in indices_per_class}
        
        # Buat batch
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Ambil sampel dari tiap kelas
            for cls, indices in indices_per_class.items():
                cls_indices = indices[pointers[cls]:pointers[cls] + self.target_per_class]
                
                # Jika tidak cukup sampel, wrapping untuk kelas ini
                if len(cls_indices) < self.target_per_class:
                    # Reset pointer dan shuffle ulang untuk kelas ini
                    pointers[cls] = 0
                    np.random.shuffle(indices_per_class[cls])
                    
                    # Ambil sisa sampel yang dibutuhkan
                    remaining = self.target_per_class - len(cls_indices)
                    cls_indices.extend(indices_per_class[cls][:remaining])
                    pointers[cls] = remaining
                else:
                    pointers[cls] += self.target_per_class
                
                batch_indices.extend(cls_indices)
            
            # Shuffle indeks dalam batch
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self) -> int:
        """
        Mendapatkan jumlah batch dalam satu epoch.
        
        Returns:
            Jumlah batch
        """
        return self.num_batches