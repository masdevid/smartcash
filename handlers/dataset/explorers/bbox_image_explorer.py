# File: smartcash/handlers/dataset/explorers/bbox_image_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer untuk analisis ukuran gambar dan bounding box dalam dataset

import cv2
import numpy as np
import collections
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class BBoxImageExplorer(BaseExplorer):
    """
    Explorer untuk analisis ukuran gambar dan bounding box dalam dataset.
    Menggabungkan ImageSizeExplorer dan BoundingBoxExplorer.
    """
    
    def explore(self, split: str, sample_size: int = 0, mode: str = 'bbox') -> Dict[str, Any]:
        """
        Analisis ukuran gambar atau bounding box dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            mode: Jenis analisis ('bbox' atau 'image_size')
            
        Returns:
            Dict hasil analisis
        """
        if mode not in ('bbox', 'image_size'):
            raise ValueError(f"Mode harus 'bbox' atau 'image_size', bukan '{mode}'")
            
        self.logger.info(f"🔍 Analisis {mode}: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
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
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk analisis {mode}")
        
        # Lakukan analisis
        if mode == 'bbox':
            return self._analyze_bbox_statistics(labels_dir, image_files)
        else:  # mode == 'image_size'
            return self._analyze_image_sizes(images_dir, image_files)
    
    def analyze_bbox(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis bounding box dalam dataset."""
        return self.explore(split, sample_size, 'bbox')
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis ukuran gambar dalam dataset."""
        return self.explore(split, sample_size, 'image_size')
    
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
    
    def _analyze_image_sizes(self, images_dir: Path, image_files: List[Path]) -> Dict[str, Any]:
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
        size_categories = {'small': 0, 'medium': 0, 'large': 0}
        
        for (w, h), count in size_counts.items():
            if w < 640 or h < 640:
                size_categories['small'] += count
            elif w > 1280 or h > 1280:
                size_categories['large'] += count
            else:
                size_categories['medium'] += count
        
        # Rekomendasi ukuran optimal
        recommended_size = self._recommend_image_size({
            'dominant_size': f"{dominant_size[0]}x{dominant_size[1]}",
            'dominant_percentage': dominant_percentage,
            'width_stats': width_stats,
            'height_stats': height_stats
        })
        
        return {
            'dominant_size': f"{dominant_size[0]}x{dominant_size[1]}",
            'dominant_percentage': dominant_percentage,
            'width_stats': width_stats,
            'height_stats': height_stats,
            'aspect_ratio_stats': aspect_ratio_stats,
            'size_categories': size_categories,
            'total_analyzed': len(width_list),
            'recommended_size': recommended_size
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
    
    def _analyze_bbox_statistics(self, labels_dir: Path, image_files: List[Path]) -> Dict[str, Any]:
        """
        Analisis statistik bounding box dalam dataset.
        
        Args:
            labels_dir: Direktori label
            image_files: List file gambar
            
        Returns:
            Dict berisi statistik bounding box
        """
        self.logger.info("🔍 Menganalisis statistik bounding box...")
        
        # Ukuran bounding box (width dan height)
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        centers_x = []  # Posisi pusat X
        centers_y = []  # Posisi pusat Y
        
        for img_path in tqdm(image_files, desc="Menganalisis bbox"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # Format YOLO: class_id, x_center, y_center, width, height
                                # (nilai dinormalisasi)
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                widths.append(width)
                                heights.append(height)
                                areas.append(width * height)
                                aspect_ratios.append(width / height if height > 0 else 0)
                                centers_x.append(x_center)
                                centers_y.append(y_center)
                            except (ValueError, IndexError, ZeroDivisionError):
                                continue
            except Exception:
                # Skip file jika ada error
                continue
        
        # Hitung statistik
        bbox_stats = {}
        
        if widths:
            # Statistik ukuran
            bbox_stats['width'] = {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'median': np.median(widths),
                'std': np.std(widths)
            }
            
            bbox_stats['height'] = {
                'min': min(heights),
                'max': max(heights),
                'mean': np.mean(heights),
                'median': np.median(heights),
                'std': np.std(heights)
            }
            
            bbox_stats['area'] = {
                'min': min(areas),
                'max': max(areas),
                'mean': np.mean(areas),
                'median': np.median(areas),
                'std': np.std(areas)
            }
            
            bbox_stats['aspect_ratio'] = {
                'min': min(aspect_ratios),
                'max': max(aspect_ratios),
                'mean': np.mean(aspect_ratios),
                'median': np.median(aspect_ratios),
                'std': np.std(aspect_ratios)
            }
            
            # Posisi pusat
            bbox_stats['center_x'] = {
                'min': min(centers_x),
                'max': max(centers_x),
                'mean': np.mean(centers_x),
                'std': np.std(centers_x)
            }
            
            bbox_stats['center_y'] = {
                'min': min(centers_y),
                'max': max(centers_y),
                'mean': np.mean(centers_y),
                'std': np.std(centers_y)
            }
            
            # Kategori ukuran bbox (kecil, sedang, besar)
            area_categories = {
                'small': 0,    # < 0.02 (area < 2%)
                'medium': 0,   # 0.02-0.1 (2-10%)
                'large': 0,    # > 0.1 (> 10%)
                'total': len(areas)
            }
            
            for area in areas:
                if area < 0.02:
                    area_categories['small'] += 1
                elif area > 0.1:
                    area_categories['large'] += 1
                else:
                    area_categories['medium'] += 1
            
            bbox_stats['area_categories'] = area_categories
            
            # Tambahkan persentase untuk kategori
            if len(areas) > 0:
                bbox_stats['area_categories']['small_pct'] = (area_categories['small'] / len(areas)) * 100
                bbox_stats['area_categories']['medium_pct'] = (area_categories['medium'] / len(areas)) * 100
                bbox_stats['area_categories']['large_pct'] = (area_categories['large'] / len(areas)) * 100
                
            # Analisis bbox per kelas
            bbox_stats['by_class'] = self._analyze_bbox_by_class(labels_dir, image_files)
        
        return bbox_stats
    
    def _analyze_bbox_by_class(self, labels_dir: Path, image_files: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Analisis bounding box berdasarkan kelas.
        
        Args:
            labels_dir: Direktori label
            image_files: List file gambar
            
        Returns:
            Dict berisi statistik bbox per kelas
        """
        self.logger.info("🔍 Menganalisis bounding box per kelas...")
        
        # Statistik per kelas
        widths_by_class = {}
        heights_by_class = {}
        areas_by_class = {}
        aspect_ratios_by_class = {}
        centers_x_by_class = {}
        centers_y_by_class = {}
        
        for img_path in tqdm(image_files, desc="Menganalisis bbox per kelas"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # Format YOLO: class_id, x_center, y_center, width, height
                                cls_id = int(float(parts[0]))
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Dapatkan nama kelas
                                class_name = self._get_class_name(cls_id)
                                
                                # Inisialisasi list jika belum ada
                                if class_name not in widths_by_class:
                                    widths_by_class[class_name] = []
                                    heights_by_class[class_name] = []
                                    areas_by_class[class_name] = []
                                    aspect_ratios_by_class[class_name] = []
                                    centers_x_by_class[class_name] = []
                                    centers_y_by_class[class_name] = []
                                
                                # Tambahkan data
                                widths_by_class[class_name].append(width)
                                heights_by_class[class_name].append(height)
                                areas_by_class[class_name].append(width * height)
                                aspect_ratios_by_class[class_name].append(width / height if height > 0 else 0)
                                centers_x_by_class[class_name].append(x_center)
                                centers_y_by_class[class_name].append(y_center)
                            except (ValueError, IndexError, ZeroDivisionError):
                                continue
            except Exception:
                # Skip file jika ada error
                continue
        
        # Hitung statistik per kelas
        stats_by_class = {}
        
        for class_name in widths_by_class.keys():
            if not widths_by_class[class_name]:
                continue
                
            stats = {
                'count': len(widths_by_class[class_name]),
                'width': {
                    'min': min(widths_by_class[class_name]),
                    'max': max(widths_by_class[class_name]),
                    'mean': np.mean(widths_by_class[class_name]),
                    'median': np.median(widths_by_class[class_name])
                },
                'height': {
                    'min': min(heights_by_class[class_name]),
                    'max': max(heights_by_class[class_name]),
                    'mean': np.mean(heights_by_class[class_name]),
                    'median': np.median(heights_by_class[class_name])
                },
                'area': {
                    'min': min(areas_by_class[class_name]),
                    'max': max(areas_by_class[class_name]),
                    'mean': np.mean(areas_by_class[class_name]),
                    'median': np.median(areas_by_class[class_name])
                },
                'aspect_ratio': {
                    'min': min(aspect_ratios_by_class[class_name]),
                    'max': max(aspect_ratios_by_class[class_name]),
                    'mean': np.mean(aspect_ratios_by_class[class_name]),
                    'median': np.median(aspect_ratios_by_class[class_name])
                }
            }
            
            stats_by_class[class_name] = stats
        
        return stats_by_class