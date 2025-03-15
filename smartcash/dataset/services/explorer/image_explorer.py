"""
File: smartcash/dataset/services/explorer/image_explorer.py
Deskripsi: Explorer untuk analisis ukuran dan properti gambar dalam dataset
"""

import cv2
import collections
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

from smartcash.dataset.services.explorer.base_explorer import BaseExplorer


class ImageExplorer(BaseExplorer):
    """Explorer khusus untuk analisis gambar."""
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis ukuran gambar
        """
        self.logger.info(f"📐 Analisis ukuran gambar untuk split {split}")
        split_path, images_dir, labels_dir, valid = self._validate_directories(split)
        
        if not valid:
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
        
        # Dapatkan file gambar
        image_files = self._get_valid_files(images_dir, labels_dir, sample_size)
        if not image_files:
            return {'status': 'error', 'message': f"Tidak ada gambar valid ditemukan"}
        
        # Hitung frekuensi ukuran gambar
        size_counts = collections.Counter()
        width_list, height_list, aspect_ratios = [], [], []
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                size_counts[(w, h)] += 1
                width_list.append(w)
                height_list.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            except Exception:
                continue
        
        # Jika tidak ada gambar valid
        if not width_list:
            return {'status': 'error', 'message': f"Tidak ada gambar yang dapat dibaca"}
        
        # Ukuran dominan
        dominant_size = size_counts.most_common(1)[0][0]
        dominant_percentage = (size_counts[dominant_size] / len(image_files)) * 100
        
        # Statistik dimensi
        width_stats = {
            'min': min(width_list),
            'max': max(width_list),
            'mean': np.mean(width_list),
            'median': np.median(width_list),
            'std': np.std(width_list)
        }
        
        height_stats = {
            'min': min(height_list),
            'max': max(height_list),
            'mean': np.mean(height_list),
            'median': np.median(height_list),
            'std': np.std(height_list)
        }
        
        aspect_ratio_stats = {
            'min': min(aspect_ratios),
            'max': max(aspect_ratios),
            'mean': np.mean(aspect_ratios),
            'median': np.median(aspect_ratios),
            'std': np.std(aspect_ratios)
        }
        
        # Kategorikan ukuran (kecil, sedang, besar)
        size_categories = {'small': 0, 'medium': 0, 'large': 0}
        
        for (w, h), count in size_counts.items():
            if w < 640 or h < 640:
                size_categories['small'] += count
            elif w > 1280 or h > 1280:
                size_categories['large'] += count
            else:
                size_categories['medium'] += count
        
        # Rekomendasi ukuran optimal
        recommended_size = self._recommend_image_size(dominant_size, width_stats, height_stats)
        
        # Kompilasi hasil
        result = {
            'status': 'success',
            'total_analyzed': len(width_list),
            'dominant_size': f"{dominant_size[0]}x{dominant_size[1]}",
            'dominant_percentage': dominant_percentage,
            'width_stats': width_stats,
            'height_stats': height_stats,
            'aspect_ratio_stats': aspect_ratio_stats,
            'size_categories': size_categories,
            'recommended_size': recommended_size
        }
        
        self.logger.info(f"📐 Ukuran dominan: {dominant_size} ({dominant_percentage:.1f}%)\n"
                         f"📐 Ukuran optimal: {recommended_size}")
        
        return result