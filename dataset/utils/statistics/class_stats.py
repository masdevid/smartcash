"""
File: smartcash/dataset/utils/statistics/class_stats.py
Deskripsi: Utilitas untuk menganalisis statistik distribusi kelas dalam dataset
"""

import os
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
import numpy as np

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class ClassStatistics:
    """Utilitas untuk analisis statistik distribusi kelas dalam dataset."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi ClassStatistics.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("class_statistics")
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        self.logger.info(f"📊 ClassStatistics diinisialisasi dengan data_dir: {self.data_dir}")
    
    def analyze_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dictionary dengan hasil analisis distribusi
        """
        self.logger.info(f"📊 Analisis distribusi kelas untuk split {split}")
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # Validasi direktori
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Direktori tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
            
        # Cari semua file gambar valid
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'error', 'message': f"Tidak ada gambar ditemukan"}
            
        # Ambil sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
            
        # Analisis distribusi kelas
        return self._calculate_distribution(image_files, labels_dir)
    
    def get_class_weights(self, split: str, method: str = 'inverse_freq') -> Dict[int, float]:
        """
        Hitung bobot kelas untuk mengatasi ketidakseimbangan.
        
        Args:
            split: Split dataset
            method: Metode pembobotan ('inverse_freq', 'sqrt_inverse', 'balanced')
            
        Returns:
            Dictionary dengan class_id: weight
        """
        result = self.analyze_distribution(split)
        if result['status'] != 'success':
            return {}
            
        counts = result['counts']
        total_objects = result['total_objects']
        num_classes = len(counts)
        
        # Konversi nama kelas ke id kelas
        class_to_id = {}
        for cls_id, class_name in self.utils.class_to_name.items():
            class_to_id[class_name] = cls_id
            
        weights = {}
        
        if method == 'inverse_freq':
            # Frekuensi invers: w_i = N / (n_classes * n_i)
            for cls, count in counts.items():
                if cls in class_to_id:
                    cls_id = class_to_id[cls]
                    if count > 0:
                        weights[cls_id] = total_objects / (num_classes * count)
                    else:
                        weights[cls_id] = 1.0
                        
        elif method == 'sqrt_inverse':
            # Sqrt frekuensi invers: w_i = sqrt(N / n_i)
            for cls, count in counts.items():
                if cls in class_to_id:
                    cls_id = class_to_id[cls]
                    if count > 0:
                        weights[cls_id] = np.sqrt(total_objects / count)
                    else:
                        weights[cls_id] = 1.0
                        
        elif method == 'balanced':
            # Bobot seimbang: w_i = 1 / n_classes
            for cls in counts:
                if cls in class_to_id:
                    cls_id = class_to_id[cls]
                    weights[cls_id] = 1.0 / num_classes
        else:
            # Default ke inverse_freq
            for cls, count in counts.items():
                if cls in class_to_id:
                    cls_id = class_to_id[cls]
                    if count > 0:
                        weights[cls_id] = total_objects / (num_classes * count)
                    else:
                        weights[cls_id] = 1.0
        
        # Normalisasi bobot
        if weights:
            max_weight = max(weights.values())
            for cls_id in weights:
                weights[cls_id] /= max_weight
                
        self.logger.info(f"⚖️ Bobot kelas menggunakan metode '{method}' dihitung untuk {len(weights)} kelas")
        return weights
        
    def _calculate_distribution(self, image_files: List[Path], labels_dir: Path) -> Dict[str, Any]:
        """
        Hitung distribusi kelas dari file yang diberikan.
        
        Args:
            image_files: Daftar file gambar
            labels_dir: Direktori label
            
        Returns:
            Dictionary dengan hasil analisis distribusi
        """
        # Hitung frekuensi kelas
        class_counts = collections.Counter()
        image_class_counts = collections.defaultdict(set)  # Kelas -> set img_stems
        
        for img_path in tqdm(image_files, desc="📊 Menganalisis distribusi kelas", unit="img"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Parse label dan hitung kelas
            bbox_data = self.utils.parse_yolo_label(label_path)
            for box in bbox_data:
                cls_id = box['class_id']
                if 'class_name' in box:
                    class_name = box['class_name']
                else:
                    class_name = self.utils.get_class_name(cls_id)
                
                class_counts[class_name] += 1
                image_class_counts[class_name].add(img_path.stem)
        
        # Statistik distribusi
        total_objects = sum(class_counts.values())
        class_percentages = {}
        images_per_class = {}
        
        if total_objects > 0:
            for cls, count in class_counts.items():
                class_percentages[cls] = (count / total_objects) * 100
                
            for cls, img_stems in image_class_counts.items():
                images_per_class[cls] = len(img_stems)
            
            # Hitung ketidakseimbangan
            if len(class_counts) > 1:
                counts = list(class_counts.values())
                max_count, min_count = max(counts), min(counts)
                # Skor antara 0-10, di mana 10 = sangat tidak seimbang
                imbalance_score = min(10.0, (max_count / max(min_count, 1) - 1) / 2)
                
                # Identifikasi kelas-kelas (under/over-represented)
                avg_count = total_objects / len(class_counts)
                underrepresented = [cls for cls, count in class_counts.items() if count < avg_count * 0.5]
                overrepresented = [cls for cls, count in class_counts.items() if count > avg_count * 2]
            else:
                imbalance_score = 0.0
                underrepresented, overrepresented = [], []
        else:
            class_percentages = {}
            images_per_class = {}
            imbalance_score = 0.0
            underrepresented, overrepresented = [], []
        
        # Log hasil
        self.logger.info(
            f"📊 Distribusi kelas:\n"
            f"   • Total objek: {total_objects}\n"
            f"   • Jumlah kelas: {len(class_counts)}\n"
            f"   • Skor ketidakseimbangan: {imbalance_score:.2f}/10"
        )
        
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_classes:
            self.logger.info("   • Top 5 kelas:")
            for cls, count in top_classes:
                percentage = class_percentages.get(cls, 0)
                self.logger.info(f"     - {cls}: {count} ({percentage:.1f}%)")
        
        # Format hasil
        result = {
            'status': 'success',
            'total_objects': total_objects,
            'class_count': len(class_counts),
            'counts': dict(class_counts),
            'percentages': class_percentages,
            'images_per_class': images_per_class,
            'imbalance_score': imbalance_score,
            'underrepresented_classes': underrepresented,
            'overrepresented_classes': overrepresented
        }
        
        return result