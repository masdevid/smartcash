"""
File: smartcash/dataset/components/samplers/weighted_sampler.py
Deskripsi: Implementasi weighted random sampler yang disesuaikan untuk dataset deteksi objek
"""

import numpy as np
from typing import Dict, List, Optional, Any, Iterator, Callable, Union
import torch
from torch.utils.data import Sampler, WeightedRandomSampler as TorchWeightedSampler

from smartcash.common.logger import get_logger


class WeightedSampler(Sampler):
    """
    Sampler yang menggunakan pembobotan berdasarkan distribusi kelas.
    Memberikan peluang sampling lebih tinggi untuk kelas yang kurang terwakili.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        class_weights: Dict[int, float] = None,
        class_getter: Callable[[int], Union[int, List[int]]] = None,
        replacement: bool = True,
        num_samples: Optional[int] = None,
        logger = None
    ):
        """
        Inisialisasi WeightedSampler.
        
        Args:
            dataset: Dataset untuk sampling
            class_weights: Bobot per kelas, {class_id: weight} (opsional)
            class_getter: Fungsi untuk mendapatkan kelas dari indeks dataset (opsional)
            replacement: Apakah sampling dengan penggantian
            num_samples: Jumlah sampel yang akan dihasilkan dalam satu epoch (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.class_weights = class_weights or {}
        self.class_getter = class_getter
        self.replacement = replacement
        self.logger = logger or get_logger("weighted_sampler")
        
        # Hitung bobot sampling untuk setiap sampel
        self.sample_weights = self._compute_sample_weights()
        
        # Jumlah sampel default = ukuran dataset
        self.num_samples = num_samples or len(dataset)
        
        # Buat torch WeightedRandomSampler
        self.torch_sampler = TorchWeightedSampler(
            weights=self.sample_weights,
            num_samples=self.num_samples,
            replacement=self.replacement
        )
        
        self.logger.info(
            f"⚖️ WeightedSampler siap dengan {len(self.class_weights)} kelas, "
            f"{self.num_samples} sampel per epoch"
        )
    
    def _compute_sample_weights(self) -> torch.Tensor:
        """
        Hitung bobot sampling untuk setiap sampel berdasarkan kelasnya.
        
        Returns:
            Tensor bobot sampling untuk setiap sampel
        """
        # Jika class_getter tidak disediakan, gunakan bobot seragam
        if self.class_getter is None or not self.class_weights:
            return torch.ones(len(self.dataset))
        
        weights = []
        
        for idx in range(len(self.dataset)):
            try:
                # Dapatkan kelas sampel
                classes = self.class_getter(idx)
                
                if isinstance(classes, (list, tuple)):
                    # Multi-kelas: ambil bobot maksimum
                    max_weight = max([self.class_weights.get(cls, 1.0) for cls in classes])
                    weights.append(max_weight)
                else:
                    # Single kelas
                    weights.append(self.class_weights.get(classes, 1.0))
            except Exception:
                # Default ke bobot 1.0 jika ada error
                weights.append(1.0)
        
        return torch.as_tensor(weights, dtype=torch.float)
    
    def __iter__(self) -> Iterator[int]:
        """
        Iterasi sampel berdasarkan bobot.
        
        Returns:
            Iterator indeks sampel
        """
        return iter(self.torch_sampler)
    
    def __len__(self) -> int:
        """
        Mendapatkan jumlah sampel dalam satu epoch.
        
        Returns:
            Jumlah sampel
        """
        return self.num_samples
    
    @classmethod
    def from_class_counts(
        cls,
        dataset: Any,
        class_counts: Dict[int, int],
        class_getter: Callable[[int], Union[int, List[int]]],
        inverse_frequency: bool = True,
        smooth_factor: float = 0.0,
        normalize: bool = True,
        **kwargs
    ) -> 'WeightedSampler':
        """
        Buat WeightedSampler dari hitungan kelas.
        
        Args:
            dataset: Dataset untuk sampling
            class_counts: Dictionary berisi {class_id: count}
            class_getter: Fungsi untuk mendapatkan kelas dari indeks dataset
            inverse_frequency: Apakah menggunakan frekuensi invers
            smooth_factor: Faktor smoothing (0 = tanpa smoothing)
            normalize: Apakah menormalisasi bobot agar rata-rata = 1.0
            **kwargs: Parameter tambahan untuk WeightedSampler
            
        Returns:
            Instance WeightedSampler
        """
        logger = kwargs.get('logger') or get_logger("weighted_sampler")
        
        # Hitung bobot berdasarkan hitungan kelas
        if not class_counts:
            logger.warning("⚠️ class_counts kosong, menggunakan bobot seragam")
            return cls(dataset, **kwargs)
        
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        class_weights = {}
        for cls_id, count in class_counts.items():
            if inverse_frequency:
                # Frekuensi invers dengan smoothing
                class_weights[cls_id] = 1.0 / (count + smooth_factor * total_samples / num_classes)
            else:
                # Bobot langsung
                class_weights[cls_id] = count / total_samples
        
        # Normalisasi bobot agar rata-rata = 1.0
        if normalize:
            mean_weight = sum(class_weights.values()) / num_classes
            class_weights = {cls: w / mean_weight for cls, w in class_weights.items()}
        
        logger.info(f"⚖️ Bobot kelas:")
        for cls_id, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   • Kelas {cls_id}: {weight:.4f}")
        
        return cls(dataset, class_weights, class_getter, **kwargs)