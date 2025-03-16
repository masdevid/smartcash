"""
File: smartcash/dataset/services/balancer/undersampler.py
Deskripsi: Implementasi berbagai metode undersampling untuk menyeimbangkan dataset
"""

import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any

from smartcash.common.logger import get_logger


class Undersampler:
    """
    Komponen untuk undersampling dataset.
    Mendukung berbagai strategi untuk mengurangi jumlah sampel kelas mayoritas.
    """
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi Undersampler.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("undersampler")
        
        self.logger.info(f"⬇️ Undersampler diinisialisasi untuk menyeimbangkan dataset")
    
    def undersample(
        self,
        data: List[Any],
        labels: List[Any],
        strategy: str = 'random',
        target_count: Optional[int] = None,
        class_list: Optional[List[Any]] = None,
        random_state: int = 42
    ) -> Tuple[List[Any], List[Any]]:
        """
        Lakukan undersampling pada dataset.
        
        Args:
            data: List data yang akan diundersampling
            labels: List label yang bersesuaian dengan data
            strategy: Strategi undersampling ('random', 'cluster', 'neighbour', 'tomek')
            target_count: Jumlah sampel target per kelas (opsional)
            class_list: Daftar kelas yang akan diundersampling (opsional)
            random_state: Seed untuk random
            
        Returns:
            Tuple (data_undersampled, labels_undersampled)
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
            self.logger.warning(f"⚠️ Tidak ada kelas yang valid untuk undersampling")
            return data, labels
        
        # Tentukan target count jika tidak ada
        if target_count is None:
            target_count = min(class_counts.values())
            
        self.logger.info(
            f"⬇️ Undersampling dengan strategi '{strategy}':\n"
            f"   • Target count: {target_count}\n"
            f"   • Distribusi awal: {len(data)} sampel, {len(class_counts)} kelas"
        )
        
        # Buat indeks per kelas
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if class_list is None or label in class_list:
                class_indices[label].append(i)
        
        # Lakukan undersampling berdasarkan strategi
        selected_indices = []
        
        # Lakukan undersampling untuk setiap kelas
        for cls, indices in class_indices.items():
            count = len(indices)
            
            # Skip kelas yang jumlahnya sudah <= target
            if count <= target_count:
                selected_indices.extend(indices)
                self.logger.info(f"   • Kelas {cls}: {count} ≤ {target_count}, tidak di-undersample")
                continue
                
            # Pilih indeks berdasarkan strategi
            selected = self._select_by_strategy(data, indices, strategy, target_count, random_state)
            selected_indices.extend(selected)
            
            self.logger.info(f"   • Kelas {cls}: {count} → {len(selected)} (-{count - len(selected)})")
        
        # Pilih data berdasarkan indeks
        data_undersampled = [data[i] for i in selected_indices]
        labels_undersampled = [labels[i] for i in selected_indices]
        
        self.logger.success(
            f"✅ Undersampling selesai:\n"
            f"   • Sampel awal: {len(data)}\n"
            f"   • Sampel akhir: {len(data_undersampled)}\n"
            f"   • Rasio: {len(data_undersampled) / len(data):.2f}"
        )
        
        return data_undersampled, labels_undersampled
    
    def _select_by_strategy(
        self,
        data: List[Any],
        indices: List[int],
        strategy: str,
        target_count: int,
        random_state: int
    ) -> List[int]:
        """
        Pilih indeks berdasarkan strategi undersampling.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            strategy: Strategi undersampling
            target_count: Jumlah sampel target
            random_state: Seed untuk random
            
        Returns:
            List indeks yang terpilih
        """
        # Jangan undersample jika jumlah sudah <= target
        if len(indices) <= target_count:
            return indices
            
        # Pilih berdasarkan strategi
        if strategy == 'random':
            # Random undersampling sederhana
            return random.sample(indices, target_count)
            
        elif strategy == 'cluster':
            # Cluster-based undersampling
            return self._cluster_undersampling(data, indices, target_count, random_state)
            
        elif strategy == 'neighbour' or strategy == 'neighbor':
            # Nearest neighbour undersampling
            return self._neighbour_undersampling(data, indices, target_count, random_state)
            
        elif strategy == 'tomek':
            # Tomek links
            return self._tomek_undersampling(data, indices, target_count, random_state)
            
        else:
            # Default ke random
            self.logger.warning(f"⚠️ Strategi '{strategy}' tidak dikenal, menggunakan 'random'")
            return random.sample(indices, target_count)
    
    def _cluster_undersampling(
        self,
        data: List[Any],
        indices: List[int],
        target_count: int,
        random_state: int
    ) -> List[int]:
        """
        Cluster-based undersampling.
        Menggunakan K-means clustering dan mengambil sampel dari setiap cluster.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            target_count: Jumlah sampel target
            random_state: Seed untuk random
            
        Returns:
            List indeks yang terpilih
        """
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            
            # Cek apakah data bisa diproses
            if not hasattr(data[0], 'shape') and not isinstance(data[0], (list, tuple, np.ndarray)):
                self.logger.warning("⚠️ Data tidak bisa di-cluster, falling back ke random sampling")
                return random.sample(indices, target_count)
            
            # Extract data untuk indeks yang dipilih
            X = np.array([data[i] for i in indices])
            
            # Flatten jika multidimensi
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            
            # Jumlah cluster
            n_clusters = min(target_count, len(indices) // 2)
            
            # Lakukan clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(X)
            
            # Pilih sampel dari setiap cluster
            selected = []
            samples_per_cluster = target_count // n_clusters
            remainder = target_count % n_clusters
            
            for cluster_id in range(n_clusters):
                cluster_indices = [indices[i] for i in range(len(indices)) if cluster_labels[i] == cluster_id]
                
                # Jumlah sampel untuk cluster ini
                samples_to_select = samples_per_cluster
                if cluster_id < remainder:
                    samples_to_select += 1
                    
                # Pilih sampel dari cluster
                if len(cluster_indices) <= samples_to_select:
                    selected.extend(cluster_indices)
                else:
                    selected.extend(random.sample(cluster_indices, samples_to_select))
            
            return selected
        
        except (ImportError, Exception) as e:
            self.logger.warning(f"⚠️ Error saat cluster undersampling: {str(e)}, falling back ke random")
            return random.sample(indices, target_count)
    
    def _neighbour_undersampling(
        self,
        data: List[Any],
        indices: List[int],
        target_count: int,
        random_state: int
    ) -> List[int]:
        """
        Nearest neighbours undersampling.
        Menggunakan NearMiss algoritma.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            target_count: Jumlah sampel target
            random_state: Seed untuk random
            
        Returns:
            List indeks yang terpilih
        """
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            
            # Cek apakah data bisa diproses
            if not hasattr(data[0], 'shape') and not isinstance(data[0], (list, tuple, np.ndarray)):
                self.logger.warning("⚠️ Data tidak bisa diproses, falling back ke random sampling")
                return random.sample(indices, target_count)
            
            # Extract data untuk indeks yang dipilih
            X = np.array([data[i] for i in indices])
            
            # Flatten jika multidimensi
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            
            # Hitung average distance ke 3 nearest neighbours untuk setiap sampel
            n_neighbors = min(4, len(X))  # 3 neighbors + self
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(X)
            
            distances, neighbors = nn.kneighbors(X)
            
            # Hitung average distance (exclude self)
            avg_distances = np.mean(distances[:, 1:], axis=1)
            
            # Sortir berdasarkan average distance (pilih yang memiliki tetangga terdekat)
            sorted_idx = np.argsort(avg_distances)
            
            # Pilih sampel dengan tetangga terdekat (lebih representatif)
            selected_local_idx = sorted_idx[:target_count]
            
            # Konversi kembali ke indeks global
            selected = [indices[i] for i in selected_local_idx]
            
            return selected
        
        except (ImportError, Exception) as e:
            self.logger.warning(f"⚠️ Error saat neighbour undersampling: {str(e)}, falling back ke random")
            return random.sample(indices, target_count)
    
    def _tomek_undersampling(
        self,
        data: List[Any],
        indices: List[int],
        target_count: int,
        random_state: int
    ) -> List[int]:
        """
        Tomek links undersampling.
        Menghapus sampel yang membentuk Tomek links.
        
        Args:
            data: List data
            indices: List indeks untuk kelas tertentu
            target_count: Jumlah sampel target
            random_state: Seed untuk random
            
        Returns:
            List indeks yang terpilih
        """
        # Jika jumlah sampel sudah <= target, langsung return semua
        if len(indices) <= target_count:
            return indices
            
        try:
            # Strategi Tomek links biasanya memerlukan label juga
            # Tanpa label kelas lain, kita tidak bisa identify Tomek links
            # Oleh karena itu, kita gunakan pendekatan cluster + random
            
            # Lakukan clustering dulu
            cluster_indices = self._cluster_undersampling(data, indices, 
                                                         min(len(indices) - 10, target_count * 2), 
                                                         random_state)
            
            # Jika masih terlalu banyak, lakukan random sampling
            if len(cluster_indices) > target_count:
                return random.sample(cluster_indices, target_count)
            else:
                return cluster_indices
                
        except (ImportError, Exception) as e:
            self.logger.warning(f"⚠️ Error saat tomek undersampling: {str(e)}, falling back ke random")
            return random.sample(indices, target_count)