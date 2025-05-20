"""
File: smartcash/dataset/services/balancer/oversampler.py
Deskripsi: Implementasi berbagai metode oversampling untuk menyeimbangkan dataset
"""

import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

from smartcash.common.logger import get_logger


class Oversampler:
    """
    Komponen untuk oversampling dataset.
    Mendukung berbagai strategi untuk menambah jumlah sampel kelas minoritas.
    """
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi Oversampler.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger()
        
        self.logger.info(f"⬆️ Oversampler diinisialisasi untuk menyeimbangkan dataset")
    
    def oversample(
        self,
        data: List[Any],
        labels: List[Any],
        strategy: str = 'duplicate',
        target_count: Optional[int] = None,
        class_list: Optional[List[Any]] = None,
        aug_pipeline = None,
        random_state: int = 42
    ) -> Tuple[List[Any], List[Any]]:
        """
        Lakukan oversampling pada dataset.
        
        Args:
            data: List data yang akan dioversample
            labels: List label yang bersesuaian dengan data
            strategy: Strategi oversampling ('duplicate', 'smote', 'adasyn', 'augmentation')
            target_count: Jumlah sampel target per kelas (opsional)
            class_list: Daftar kelas yang akan dioversample (opsional)
            aug_pipeline: Pipeline augmentasi untuk strategy 'augmentation'
            random_state: Seed untuk random
            
        Returns:
            Tuple (data_oversampled, labels_oversampled)
        """
        # Setup random state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Validasi input
        if len(data) != len(labels):
            self.logger.error(f"❌ Jumlah data dan label tidak sama: {len(data)} != {len(labels)}")
            return data, labels
        
        if not data:
            return [], []
        
        # Hitung distribusi kelas
        counter = Counter(labels)
        class_counts = dict(counter)
        
        # Terapkan filter kelas jika ada
        if class_list:
            # Filter hanya kelas yang diminta
            class_counts = {cls: count for cls, count in class_counts.items() if cls in class_list}
            
        if not class_counts:
            self.logger.warning(f"⚠️ Tidak ada kelas yang valid untuk oversampling")
            return data, labels
        
        # Tentukan target count jika tidak ada
        if target_count is None:
            target_count = max(class_counts.values())
            
        self.logger.info(
            f"⬆️ Oversampling dengan strategi '{strategy}':\n"
            f"   • Target count: {target_count}\n"
            f"   • Distribusi awal: {len(data)} sampel, {len(class_counts)} kelas"
        )
        
        # Buat indeks per kelas
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if class_list is None or label in class_list:
                class_indices[label].append(i)
        
        # Buat data dan label baru (salin data lama dulu)
        new_data = data.copy()
        new_labels = labels.copy()
        
        # Lakukan oversampling untuk setiap kelas
        for cls, indices in class_indices.items():
            count = len(indices)
            
            # Skip kelas yang jumlahnya sudah >= target
            if count >= target_count:
                self.logger.info(f"   • Kelas {cls}: {count} ≥ {target_count}, tidak di-oversample")
                continue
                
            # Hitung jumlah sampel baru yang diperlukan
            n_samples_needed = target_count - count
            
            # Generate sampel baru berdasarkan strategi
            new_samples = self._generate_by_strategy(
                data, indices, strategy, n_samples_needed, aug_pipeline, random_state
            )
            
            # Tambahkan sampel baru ke dataset
            new_data.extend(new_samples)
            new_labels.extend([cls] * len(new_samples))
            
            self.logger.info(f"   • Kelas {cls}: {count} → {count + len(new_samples)} (+{len(new_samples)})")
        
        self.logger.success(
            f"✅ Oversampling selesai:\n"
            f"   • Sampel awal: {len(data)}\n"
            f"   • Sampel akhir: {len(new_data)}\n"
            f"   • Rasio: {len(new_data) / len(data):.2f}"
        )
        
        return new_data, new_labels
    
    def _generate_by_strategy(
        self,
        data: List[Any],
        indices: List[int],
        strategy: str,
        n_samples: int,
        aug_pipeline,
        random_state: int
    ) -> List[Any]:
        """
        Generate sampel baru berdasarkan strategi oversampling.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            strategy: Strategi oversampling
            n_samples: Jumlah sampel baru yang diperlukan
            aug_pipeline: Pipeline augmentasi (jika strategy 'augmentation')
            random_state: Seed untuk random
            
        Returns:
            List sampel baru
        """
        # Pilih berdasarkan strategi
        if strategy == 'duplicate':
            # Random duplicate
            return self._duplicate_samples(data, indices, n_samples)
            
        elif strategy == 'smote':
            # SMOTE - Synthetic Minority Over-sampling Technique
            return self._smote_samples(data, indices, n_samples, random_state)
            
        elif strategy == 'adasyn':
            # ADASYN - Adaptive Synthetic Sampling
            return self._adasyn_samples(data, indices, n_samples, random_state)
            
        elif strategy == 'augmentation':
            # Augmentation-based oversampling
            return self._augmentation_samples(data, indices, n_samples, aug_pipeline)
            
        else:
            # Default ke duplicate
            self.logger.warning(f"⚠️ Strategi '{strategy}' tidak dikenal, menggunakan 'duplicate'")
            return self._duplicate_samples(data, indices, n_samples)
    
    def _duplicate_samples(self, data: List[Any], indices: List[int], n_samples: int) -> List[Any]:
        """
        Generate sampel baru dengan menduplikasi sampel yang ada.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            n_samples: Jumlah sampel baru yang diperlukan
            
        Returns:
            List sampel baru
        """
        # Ambil sampel dengan replacement
        sampled_indices = random.choices(indices, k=n_samples)
        return [data[i] for i in sampled_indices]
    
    def _smote_samples(self, data: List[Any], indices: List[int], n_samples: int, random_state: int) -> List[Any]:
        """
        Generate sampel baru dengan SMOTE (Synthetic Minority Over-sampling Technique).
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            n_samples: Jumlah sampel baru yang diperlukan
            random_state: Seed untuk random
            
        Returns:
            List sampel baru
        """
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            
            # Cek apakah data bisa diproses
            if not hasattr(data[0], 'shape') and not isinstance(data[0], (list, tuple, np.ndarray)):
                self.logger.warning("⚠️ Data tidak bisa diproses dengan SMOTE, falling back ke duplicate sampling")
                return self._duplicate_samples(data, indices, n_samples)
            
            # Extract data untuk indeks yang dipilih
            X = np.array([data[i] for i in indices])
            
            # Flatten jika multidimensi
            if len(X.shape) > 2:
                original_shape = X.shape[1:]
                X = X.reshape(X.shape[0], -1)
            else:
                original_shape = None
            
            # SMOTE membutuhkan minimal 2 sampel dan k+1 tetangga
            if len(X) < 2:
                self.logger.warning("⚠️ Terlalu sedikit sampel untuk SMOTE, falling back ke duplicate")
                return self._duplicate_samples(data, indices, n_samples)
            
            # Setup nearest neighbors
            k = min(5, len(X) - 1)  # k tetangga
            nn = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, neighbors = nn.kneighbors(X)
            
            # Generate sampel SMOTE
            synthetic_samples = []
            
            for _ in range(n_samples):
                # Pilih random sampel
                idx = random.randint(0, len(X) - 1)
                
                # Pilih random tetangga (skip self)
                nn_idx = random.choice(neighbors[idx][1:])
                
                # Hitung sampel sintetis
                alpha = random.random()
                sample = X[idx] + alpha * (X[nn_idx] - X[idx])
                
                # Reshape kembali jika perlu
                if original_shape:
                    sample = sample.reshape(original_shape)
                
                synthetic_samples.append(sample)
            
            return synthetic_samples
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"⚠️ Error saat generate dengan SMOTE: {str(e)}, falling back ke duplicate")
            return self._duplicate_samples(data, indices, n_samples)
    
    def _adasyn_samples(self, data: List[Any], indices: List[int], n_samples: int, random_state: int) -> List[Any]:
        """
        Generate sampel baru dengan ADASYN (Adaptive Synthetic Sampling).
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            n_samples: Jumlah sampel baru yang diperlukan
            random_state: Seed untuk random
            
        Returns:
            List sampel baru
        """
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            
            # Cek apakah data bisa diproses
            if not hasattr(data[0], 'shape') and not isinstance(data[0], (list, tuple, np.ndarray)):
                self.logger.warning("⚠️ Data tidak bisa diproses dengan ADASYN, falling back ke duplicate sampling")
                return self._duplicate_samples(data, indices, n_samples)
            
            # Extract data untuk indeks yang dipilih
            X = np.array([data[i] for i in indices])
            
            # Flatten jika multidimensi
            if len(X.shape) > 2:
                original_shape = X.shape[1:]
                X = X.reshape(X.shape[0], -1)
            else:
                original_shape = None
            
            # ADASYN membutuhkan minimal kelas dengan sampel cukup
            if len(X) < 2:
                self.logger.warning("⚠️ Terlalu sedikit sampel untuk ADASYN, falling back ke duplicate")
                return self._duplicate_samples(data, indices, n_samples)
            
            # Setup nearest neighbors
            k = min(5, len(X) - 1)  # k tetangga
            nn = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, neighbors = nn.kneighbors(X)
            
            # Generate sampel ADASYN
            synthetic_samples = []
            
            for _ in range(n_samples):
                # Pilih random sampel
                idx = random.randint(0, len(X) - 1)
                
                # Pilih random tetangga (skip self)
                nn_idx = random.choice(neighbors[idx][1:])
                
                # Hitung sampel sintetis
                alpha = random.random()
                sample = X[idx] + alpha * (X[nn_idx] - X[idx])
                
                # Reshape kembali jika perlu
                if original_shape:
                    sample = sample.reshape(original_shape)
                
                synthetic_samples.append(sample)
            
            return synthetic_samples
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"⚠️ Error saat generate dengan ADASYN: {str(e)}, falling back ke duplicate")
            return self._duplicate_samples(data, indices, n_samples)
    
    def _augmentation_samples(
        self, 
        data: List[Any], 
        indices: List[int], 
        n_samples: int, 
        aug_pipeline
    ) -> List[Any]:
        """
        Generate sampel baru dengan data augmentation.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            n_samples: Jumlah sampel baru yang diperlukan
            aug_pipeline: Pipeline augmentasi untuk generate sampel baru
            
        Returns:
            List sampel baru
        """
        if aug_pipeline is None:
            self.logger.warning("⚠️ Tidak ada pipeline augmentasi, falling back ke duplicate")
            return self._duplicate_samples(data, indices, n_samples)
            
        try:
            # Pilih indeks dengan replacement
            sampled_indices = random.choices(indices, k=n_samples)
            
            # Generate sampel baru dengan augmentasi
            new_samples = []
            
            for idx in sampled_indices:
                original = data[idx]
                
                if hasattr(aug_pipeline, 'transform'):
                    # Albumentations style
                    if hasattr(original, 'shape'):
                        # Untuk gambar
                        augmented = aug_pipeline(image=original)['image']
                    else:
                        # Untuk objek lain, skip
                        augmented = original
                    
                else:
                    # Custom transform function
                    try:
                        augmented = aug_pipeline(original)
                    except Exception:
                        augmented = original
                
                new_samples.append(augmented)
                
            return new_samples
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat augmentasi: {str(e)}, falling back ke duplicate")
            return self._duplicate_samples(data, indices, n_samples)