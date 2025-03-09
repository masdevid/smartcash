# File: smartcash/handlers/dataset/explorers/image_size_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer khusus untuk analisis ukuran gambar dalam dataset

import collections
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class ImageSizeExplorer(BaseExplorer):
    """
    Explorer khusus untuk analisis ukuran gambar dalam dataset.
    Menganalisis resolusi, aspek rasio, dan distribusi ukuran.
    """
    
    def explore(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis ukuran gambar
        """
        self.logger.info(f"🔍 Analisis ukuran gambar: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        
        if not images_dir.exists():
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar di split {split}")
            return {'error': f"Tidak ada file gambar di split {split}"}
        
        # Batasi sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk analisis ukuran")
        
        # Analisis ukuran gambar
        image_size_stats = self._analyze_image_sizes(images_dir, image_files)
        
        # Log hasil analisis
        total_analyzed = image_size_stats.get('total_analyzed', 0)
        dominant_size = image_size_stats.get('dominant_size', 'N/A')
        dominant_percentage = image_size_stats.get('dominant_percentage', 0)
        size_categories = image_size_stats.get('size_categories', {})
        
        self.logger.info(
            f"📊 Hasil analisis ukuran gambar '{split}':\n"
            f"   • Total gambar dianalisis: {total_analyzed}\n"
            f"   • Ukuran dominan: {dominant_size} ({dominant_percentage:.1f}%)\n"
            f"   • Gambar kecil (<640x640): {size_categories.get('small', 0)}\n"
            f"   • Gambar sedang (640-1280): {size_categories.get('medium', 0)}\n"
            f"   • Gambar besar (>1280x1280): {size_categories.get('large', 0)}"
        )
        
        # Rekomendasi ukuran
        recommended_size = self._recommend_image_size(image_size_stats)
        image_size_stats['recommended_size'] = recommended_size
        
        self.logger.info(f"💡 Rekomendasi ukuran gambar: {recommended_size}")
        
        return image_size_stats
    
    def get_dataset_sizes(self, split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Dapatkan statistik ukuran untuk seluruh dataset atau split tertentu.
        
        Args:
            split: Split dataset tertentu (opsional, jika None semua split dianalisis)
            
        Returns:
            Dict berisi statistik ukuran untuk setiap split
        """
        self.logger.info(f"📊 Menganalisis ukuran gambar dalam dataset")
        
        splits = [split] if split else ['train', 'valid', 'test']
        size_stats = {}
        
        for current_split in splits:
            split_dir = self._get_split_path(current_split)
            images_dir = split_dir / 'images'
            
            if not images_dir.exists():
                size_stats[current_split] = {'error': f"Split {current_split} tidak ditemukan"}
                continue
                
            # Cari semua file gambar
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(images_dir.glob(ext)))
                
            if not image_files:
                size_stats[current_split] = {'error': f"Tidak ada file gambar di split {current_split}"}
                continue
                
            # Analisis ukuran gambar
            size_stats[current_split] = self._analyze_image_sizes(images_dir, image_files)
        
        return size_stats
    
    def _analyze_image_sizes(
        self,
        images_dir: Path,
        image_files: List[Path]
    ) -> Dict[str, Any]:
        """
        Analisis distribusi ukuran gambar dalam dataset.
        
        Args:
            images_dir: Direktori gambar
            image_files: List file gambar
            
        Returns:
            Dict berisi statistik ukuran gambar
        """
        self.logger.info("🔍 Menganalisis ukuran gambar...")
        
        # Hitung frekuensi ukuran gambar
        size_counts = collections.Counter()
        width_list = []
        height_list = []
        aspect_ratios = []
        
        for img_path in tqdm(image_files, desc="Menganalisis ukuran"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    size_counts[(w, h)] += 1
                    width_list.append(w)
                    height_list.append(h)
                    aspect_ratios.append(w / h if h > 0 else 0)
            except Exception:
                # Skip file jika ada error
                continue
        
        # Ukuran dominan
        dominant_size = size_counts.most_common(1)[0][0] if size_counts else (0, 0)
        dominant_percentage = (size_counts[dominant_size] / len(image_files)) * 100 if image_files else 0
        
        # Statistik dimensi
        width_stats = {
            'min': min(width_list) if width_list else 0,
            'max': max(width_list) if width_list else 0,
            'mean': np.mean(width_list) if width_list else 0,
            'median': np.median(width_list) if width_list else 0
        }
        
        height_stats = {
            'min': min(height_list) if height_list else 0,
            'max': max(height_list) if height_list else 0,
            'mean': np.mean(height_list) if height_list else 0,
            'median': np.median(height_list) if height_list else 0
        }
        
        aspect_ratio_stats = {
            'min': min(aspect_ratios) if aspect_ratios else 0,
            'max': max(aspect_ratios) if aspect_ratios else 0,
            'mean': np.mean(aspect_ratios) if aspect_ratios else 0,
            'median': np.median(aspect_ratios) if aspect_ratios else 0
        }
        
        # Kategorikan ukuran (kecil, sedang, besar)
        size_categories = {
            'small': 0,   # < 640 x 640
            'medium': 0,  # 640-1280 x 640-1280
            'large': 0    # > 1280 x 1280
        }
        
        for (w, h), count in size_counts.items():
            if w < 640 or h < 640:
                size_categories['small'] += count
            elif w > 1280 or h > 1280:
                size_categories['large'] += count
            else:
                size_categories['medium'] += count
        
        return {
            'dominant_size': f"{dominant_size[0]}x{dominant_size[1]}",
            'dominant_percentage': dominant_percentage,
            'width_stats': width_stats,
            'height_stats': height_stats,
            'aspect_ratio_stats': aspect_ratio_stats,
            'size_categories': size_categories,
            'total_analyzed': len(width_list)
        }
    
    def _recommend_image_size(self, image_stats: Dict[str, Any]) -> Tuple[int, int]:
        """
        Berikan rekomendasi ukuran gambar optimal berdasarkan statistik.
        
        Args:
            image_stats: Statistik ukuran gambar
            
        Returns:
            Tuple (width, height) ukuran yang direkomendasikan
        """
        # Ekstrak statistik terkait
        dominant_size = image_stats.get('dominant_size', '640x640')
        
        width_stats = image_stats.get('width_stats', {})
        height_stats = image_stats.get('height_stats', {})
        
        median_width = width_stats.get('median', 640)
        median_height = height_stats.get('median', 640)
        
        # Ekstrak dimensi dari dominant_size
        try:
            width, height = map(int, dominant_size.split('x'))
        except Exception:
            width, height = 640, 640
        
        # Jika dominant_size tidak cukup representatif (persentase rendah)
        dominant_percentage = image_stats.get('dominant_percentage', 0)
        if dominant_percentage < 50:
            # Gunakan median sebagai basis
            width, height = int(median_width), int(median_height)
        
        # Pilih ukuran terdekat yang kelipatan 32 (optimal untuk YOLO)
        recommended_width = int(round(width / 32) * 32)
        recommended_height = int(round(height / 32) * 32)
        
        # Pastikan minimal 320x320, maksimal 1280x1280 (batas YOLO)
        recommended_width = max(320, min(1280, recommended_width))
        recommended_height = max(320, min(1280, recommended_height))
        
        return (recommended_width, recommended_height)