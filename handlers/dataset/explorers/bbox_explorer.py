# File: smartcash/handlers/dataset/explorers/bbox_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer khusus untuk analisis bounding box dalam dataset

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class BoundingBoxExplorer(BaseExplorer):
    """
    Explorer khusus untuk analisis bounding box dalam dataset.
    Menganalisis ukuran, rasio aspek, dan posisi bounding box.
    """
    
    def explore(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis bounding box dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis bounding box
        """
        self.logger.info(f"🔍 Analisis bounding box: {split}")
        
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
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk analisis bounding box")
        
        # Analisis bbox
        bbox_stats = self._analyze_bbox_statistics(labels_dir, image_files)
        
        # Tambahkan analisis per kelas jika diperlukan
        bbox_stats_by_class = self._analyze_bbox_by_class(labels_dir, image_files)
        bbox_stats['by_class'] = bbox_stats_by_class
        
        # Log hasil analisis
        self._log_bbox_stats(split, bbox_stats)
        
        return bbox_stats
    
    def _analyze_bbox_statistics(
        self,
        labels_dir: Path,
        image_files: List[Path]
    ) -> Dict[str, Any]:
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
        
        return bbox_stats
    
    def _analyze_bbox_by_class(
        self,
        labels_dir: Path,
        image_files: List[Path]
    ) -> Dict[str, Dict[str, Any]]:
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
    
    def _log_bbox_stats(self, split: str, bbox_stats: Dict[str, Any]) -> None:
        """
        Log statistik bbox ke console.
        
        Args:
            split: Split dataset
            bbox_stats: Statistik bbox
        """
        if not bbox_stats:
            self.logger.warning(f"⚠️ Tidak ada data bounding box untuk dianalisis di split {split}")
            return
            
        # Statistik umum
        area_categories = bbox_stats.get('area_categories', {})
        total_bbox = area_categories.get('total', 0)
        
        self.logger.info(
            f"📊 Statistik bounding box di split '{split}':\n"
            f"   • Total bounding box: {total_bbox}\n"
            f"   • Ukuran kecil (<2% area): {area_categories.get('small', 0)} ({area_categories.get('small_pct', 0):.1f}%)\n"
            f"   • Ukuran sedang (2-10%): {area_categories.get('medium', 0)} ({area_categories.get('medium_pct', 0):.1f}%)\n"
            f"   • Ukuran besar (>10%): {area_categories.get('large', 0)} ({area_categories.get('large_pct', 0):.1f}%)"
        )
        
        # Width & height
        width_stats = bbox_stats.get('width', {})
        height_stats = bbox_stats.get('height', {})
        
        if width_stats and height_stats:
            self.logger.info(
                f"   • Dimensi (normalized): "
                f"Width {width_stats.get('mean', 0):.3f} x Height {height_stats.get('mean', 0):.3f} (mean)\n"
                f"   • Rasio aspek: {bbox_stats.get('aspect_ratio', {}).get('mean', 0):.3f} (mean)"
            )
        
        # Posisi pusat
        center_x = bbox_stats.get('center_x', {})
        center_y = bbox_stats.get('center_y', {})
        
        if center_x and center_y:
            self.logger.info(
                f"   • Posisi pusat (normalized): "
                f"X {center_x.get('mean', 0):.3f} ± {center_x.get('std', 0):.3f}, "
                f"Y {center_y.get('mean', 0):.3f} ± {center_y.get('std', 0):.3f}"
            )
        
        # Log statistik per kelas (top 5)
        by_class = bbox_stats.get('by_class', {})
        if by_class:
            # Sort by jumlah objek
            sorted_classes = sorted(by_class.items(), key=lambda x: x[1].get('count', 0), reverse=True)
            
            self.logger.info(f"📊 Top 5 kelas berdasarkan jumlah objek:")
            for i, (class_name, stats) in enumerate(sorted_classes[:5]):
                self.logger.info(
                    f"   {i+1}. {class_name}: {stats.get('count', 0)} objek, "
                    f"area mean: {stats.get('area', {}).get('mean', 0):.4f}"
                )