"""
File: smartcash/dataset/services/balancer/weight_calculator.py
Deskripsi: Komponen untuk menghitung bobot sampling berdasarkan distribusi kelas
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Union, Any, Tuple

from smartcash.common.logger import get_logger


class WeightCalculator:
    """
    Komponen untuk menghitung bobot sampling untuk mengatasi ketidakseimbangan kelas.
    Mendukung beberapa strategi pembobotan untuk weighted sampling.
    """
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi WeightCalculator.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger()
        
        # Setup parameter balancing
        self.balance_config = self.config.get('dataset', {}).get('balance', {})
        self.smoothing_factor = self.balance_config.get('smoothing_factor', 0.5)
        self.scale_weights = self.balance_config.get('scale_weights', 'normalize')  # 'normalize', 'min-max', 'none'
        
        self.logger.info(f"⚖️ WeightCalculator diinisialisasi untuk menghitung bobot sampling")
    
    def calculate_class_weights(
        self, 
        class_counts: Dict[str, int], 
        strategy: str = 'inverse',
        beta: float = 0.999
    ) -> Dict[str, float]:
        """
        Hitung bobot untuk setiap kelas berdasarkan distribusi.
        
        Args:
            class_counts: Dictionary berisi jumlah sampel per kelas
            strategy: Strategi perhitungan bobot ('inverse', 'sqrt_inverse', 'effective', 'log_inverse')
            beta: Parameter untuk effective number balancing
            
        Returns:
            Dictionary berisi bobot per kelas
        """
        if not class_counts:
            return {}
            
        weights = {}
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # Frekuensi kelas
        frequencies = {cls: count / total_samples for cls, count in class_counts.items()}
        
        # Efektif num samples (untuk metode effective number)
        effective_nums = {cls: (1 - beta**count) / (1 - beta) for cls, count in class_counts.items()}
        
        # Hitung bobot berdasarkan strategi
        if strategy == 'inverse':
            # Inverse frequency weighting
            weights = {cls: 1 / (freq + self.smoothing_factor) for cls, freq in frequencies.items()}
            
        elif strategy == 'sqrt_inverse':
            # Square root of inverse frequency
            weights = {cls: 1 / np.sqrt(freq + self.smoothing_factor) for cls, freq in frequencies.items()}
            
        elif strategy == 'log_inverse':
            # Log inverse frequency
            weights = {cls: 1 / np.log1p(freq * 100) for cls, freq in frequencies.items()}
            
        elif strategy == 'effective':
            # Effective number of samples (Lin et al., 2017)
            weights = {cls: 1 / (eff_num + self.smoothing_factor) for cls, eff_num in effective_nums.items()}
            
        elif strategy == 'balanced':
            # Balanced weighting (keras style)
            weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
            
        elif strategy == 'power':
            # Power-based weighting
            max_count = max(class_counts.values())
            weights = {cls: (max_count / (count + self.smoothing_factor))**0.75 for cls, count in class_counts.items()}
            
        else:
            # Default to inverse frequency
            self.logger.warning(f"⚠️ Strategi '{strategy}' tidak dikenal, menggunakan 'inverse'")
            weights = {cls: 1 / (freq + self.smoothing_factor) for cls, freq in frequencies.items()}
        
        # Skala bobot jika diminta
        if self.scale_weights == 'normalize':
            # Normalize weights to sum to 1
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                weights = {cls: w / weight_sum for cls, w in weights.items()}
                
        elif self.scale_weights == 'min-max':
            # Min-max scaling
            min_w = min(weights.values())
            max_w = max(weights.values())
            if max_w > min_w:
                weights = {cls: (w - min_w) / (max_w - min_w) * 0.9 + 0.1 for cls, w in weights.items()}
        
        # Log bobot
        self.logger.info(f"⚖️ Bobot kelas dihitung dengan metode '{strategy}':")
        for cls, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            self.logger.info(f"   • {cls}: {weight:.4f}")
        
        if len(weights) > 5:
            self.logger.info(f"   • ... dan {len(weights) - 5} kelas lainnya")
            
        return weights
    
    def calculate_sample_weights(
        self, 
        targets: List[Union[int, str]], 
        class_weights: Dict[Union[int, str], float]
    ) -> np.ndarray:
        """
        Hitung bobot untuk setiap sampel berdasarkan bobot kelas.
        
        Args:
            targets: List target kelas (class ID atau nama) untuk setiap sampel
            class_weights: Dictionary berisi bobot per kelas
            
        Returns:
            Array NumPy berisi bobot untuk setiap sampel
        """
        sample_weights = np.ones(len(targets))
        
        # Terapkan bobot untuk setiap sampel
        for i, target in enumerate(targets):
            if target in class_weights:
                sample_weights[i] = class_weights[target]
        
        return sample_weights
    
    def get_balanced_sampler_weights(
        self, 
        dataset, 
        strategy: str = 'inverse'
    ) -> Tuple[np.ndarray, Dict[Union[int, str], float]]:
        """
        Hitung bobot untuk WeightedRandomSampler pada dataset.
        
        Args:
            dataset: Dataset yang akan dihitung bobotnya
            strategy: Strategi perhitungan bobot
            
        Returns:
            Tuple (array bobot sampel, dictionary bobot kelas)
        """
        # Hitung frekuensi kelas
        class_counts = Counter()
        
        if hasattr(dataset, 'get_class_statistics'):
            # Gunakan method bawaan jika ada
            class_counts.update(dataset.get_class_statistics())
        else:
            # Hitung manual
            for i in range(len(dataset)):
                try:
                    # Coba akses label
                    _, target = dataset[i]
                    if isinstance(target, dict) and 'targets' in target:
                        # Format multilayer
                        for layer_targets in target['targets'].values():
                            for cls_id in layer_targets:
                                class_counts[cls_id] += 1
                    elif isinstance(target, list):
                        # Format flat list
                        for cls_id in target:
                            class_counts[cls_id] += 1
                    else:
                        # Format scalar
                        class_counts[target] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Error saat mengakses label: {str(e)}")
        
        # Hitung bobot kelas
        class_weights = self.calculate_class_weights(dict(class_counts), strategy=strategy)
        
        # Terapkan ke setiap sampel
        weights = np.ones(len(dataset))
        
        # Terapkan bobot untuk setiap sampel
        for i in range(len(dataset)):
            try:
                _, target = dataset[i]
                if isinstance(target, dict) and 'targets' in target:
                    # Format multilayer, ambil bobot tertinggi
                    sample_weight = 1.0
                    for layer_targets in target['targets'].values():
                        for cls_id in layer_targets:
                            if cls_id in class_weights:
                                sample_weight = max(sample_weight, class_weights[cls_id])
                    weights[i] = sample_weight
                elif isinstance(target, list):
                    # Format flat list, ambil bobot tertinggi
                    sample_weight = 1.0
                    for cls_id in target:
                        if cls_id in class_weights:
                            sample_weight = max(sample_weight, class_weights[cls_id])
                    weights[i] = sample_weight
                else:
                    # Format scalar
                    if target in class_weights:
                        weights[i] = class_weights[target]
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat menerapkan bobot: {str(e)}")
        
        return weights, class_weights
    
    def calculate_focal_loss_weights(
        self, 
        class_counts: Dict[str, int], 
        gamma: float = 2.0, 
        alpha: float = 0.25
    ) -> Dict[str, float]:
        """
        Hitung bobot untuk focal loss.
        
        Args:
            class_counts: Dictionary berisi jumlah sampel per kelas
            gamma: Parameter gamma untuk focal loss
            alpha: Parameter alpha untuk focal loss
            
        Returns:
            Dictionary berisi bobot per kelas
        """
        # Hitung frekuensi kelas
        total_samples = sum(class_counts.values())
        frequencies = {cls: count / total_samples for cls, count in class_counts.items()}
        
        # Hitung bobot dengan formula focal loss
        weights = {}
        for cls, freq in frequencies.items():
            # Bobot berbanding terbalik dengan frekuensi
            # Formula: alpha * (1 - freq)^gamma
            weights[cls] = alpha * ((1 - freq) ** gamma)
        
        # Normalisasi bobot
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {cls: w / weight_sum * len(weights) for cls, w in weights.items()}
        
        return weights
    
    def get_label_smoothing_factor(self, class_counts: Dict[str, int]) -> float:
        """
        Hitung faktor label smoothing berdasarkan distribusi kelas.
        
        Args:
            class_counts: Dictionary berisi jumlah sampel per kelas
            
        Returns:
            Faktor label smoothing yang disarankan
        """
        # Hitung imbalance ratio
        if not class_counts:
            return 0.1
            
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        if min_count == 0:
            imbalance_ratio = float('inf')
        else:
            imbalance_ratio = max_count / min_count
        
        # Tentukan label smoothing berdasarkan imbalance ratio
        if imbalance_ratio < 2:
            return 0.1  # Hampir seimbang
        elif imbalance_ratio < 5:
            return 0.2  # Sedikit tidak seimbang
        elif imbalance_ratio < 10:
            return 0.3  # Sedang tidak seimbang
        else:
            return 0.4  # Sangat tidak seimbang