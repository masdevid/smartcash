"""
File: smartcash/utils/dataset/dataset_analyzer.py
Author: Alfrida Sabar
Deskripsi: Modul untuk analisis mendalam dataset dengan metrik statistik dan visualisasi
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import random
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class DatasetAnalyzer:
    """
    Kelas untuk melakukan analisis mendalam tentang dataset.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi analyzer dataset.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
    
    def analyze_image_sizes(
        self,
        split: str,
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset
            sample_size: Jika > 0, gunakan sampel
            
        Returns:
            Dict statistik ukuran gambar
        """
        images_dir = self.data_dir / split / 'images'
        
        if not images_dir.exists():
            return {
                'status': 'error',
                'message': f"Direktori gambar tidak ditemukan: {images_dir}",
                'sizes': {}
            }
        
        # Temukan semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        if not image_files:
            return {
                'status': 'error',
                'message': f"Tidak ada gambar ditemukan di {images_dir}",
                'sizes': {}
            }
            
        # Jika sample_size ditentukan, ambil sampel acak
        if 0 < sample_size < len(image_files):
            random.seed(42)  # Untuk hasil yang konsisten
            image_files = random.sample(image_files, sample_size)
        
        # Analisis ukuran
        size_counts = {}
        aspect_ratios = {}
        width_sum = 0
        height_sum = 0
        
        for img_path in tqdm(image_files, desc="Analisis Ukuran Gambar"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                size_key = f"{w}x{h}"
                
                # Track ukuran
                if size_key not in size_counts:
                    size_counts[size_key] = 0
                size_counts[size_key] += 1
                
                # Track rasio aspek (dibulatkan ke 2 desimal)
                aspect = round(w / h, 2)
                if aspect not in aspect_ratios:
                    aspect_ratios[aspect] = 0
                aspect_ratios[aspect] += 1
                
                # Akumulasi untuk rata-rata
                width_sum += w
                height_sum += h
                
            except Exception:
                continue
        
        # Tentukan ukuran dominan
        if size_counts:
            dominant_size = max(size_counts.items(), key=lambda x: x[1])
            dominant_pct = (dominant_size[1] / len(image_files)) * 100
        else:
            dominant_size = ("Unknown", 0)
            dominant_pct = 0
            
        # Tentukan rasio aspek dominan
        if aspect_ratios:
            dominant_aspect = max(aspect_ratios.items(), key=lambda x: x[1])
            dominant_aspect_pct = (dominant_aspect[1] / len(image_files)) * 100
        else:
            dominant_aspect = (0, 0)
            dominant_aspect_pct = 0
            
        # Hitung rata-rata
        avg_width = width_sum / max(1, len(image_files))
        avg_height = height_sum / max(1, len(image_files))
        
        return {
            'status': 'success',
            'total_images': len(image_files),
            'sizes': dict(sorted(size_counts.items(), key=lambda x: x[1], reverse=True)),
            'aspect_ratios': dict(sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True)),
            'dominant_size': dominant_size[0],
            'dominant_size_count': dominant_size[1],
            'dominant_size_percent': dominant_pct,
            'dominant_aspect_ratio': dominant_aspect[0],
            'dominant_aspect_ratio_count': dominant_aspect[1],
            'dominant_aspect_ratio_percent': dominant_aspect_pct,
            'mean_width': avg_width,
            'mean_height': avg_height
        }
    
    def analyze_class_balance(self, validation_results: Dict) -> Dict:
        """
        Analisis keseimbangan kelas dalam dataset.
        
        Args:
            validation_results: Hasil validasi dataset
            
        Returns:
            Dict statistik keseimbangan kelas
        """
        class_stats = validation_results.get('class_stats', {})
        
        if not class_stats:
            return {
                'status': 'error',
                'message': "Tidak ada statistik kelas dalam hasil validasi",
                'imbalance_score': 10.0
            }
            
        # Hitung statistik dasar
        total_objects = sum(class_stats.values())
        if total_objects == 0:
            return {
                'status': 'error',
                'message': "Tidak ada objek yang valid",
                'imbalance_score': 10.0
            }
            
        # Hitung persentase per kelas
        class_percentages = {
            cls: (count / total_objects) * 100
            for cls, count in class_stats.items()
        }
        
        # Hitung ketidakseimbangan
        mean_percentage = 100 / len(class_stats)
        max_deviation = max(abs(pct - mean_percentage) for pct in class_percentages.values())
        
        # Normalisasi ketidakseimbangan ke skala 0-10
        # 0 = seimbang sempurna, 10 = sangat tidak seimbang
        imbalance_score = min(10.0, (max_deviation / mean_percentage) * 5)
        
        # Identifikasi kelas yang kurang terwakili
        underrepresented = [
            cls for cls, pct in class_percentages.items()
            if pct < mean_percentage * 0.5
        ]
        
        # Identifikasi kelas yang terlalu terwakili
        overrepresented = [
            cls for cls, pct in class_percentages.items()
            if pct > mean_percentage * 2
        ]
        
        return {
            'status': 'success',
            'total_objects': total_objects,
            'class_count': len(class_stats),
            'class_percentages': dict(sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)),
            'mean_percentage': mean_percentage,
            'max_deviation': max_deviation,
            'imbalance_score': imbalance_score,
            'underrepresented_classes': underrepresented,
            'overrepresented_classes': overrepresented
        }
    
    def analyze_layer_balance(self, validation_results: Dict) -> Dict:
        """
        Analisis keseimbangan layer dalam dataset.
        
        Args:
            validation_results: Hasil validasi dataset
            
        Returns:
            Dict statistik keseimbangan layer
        """
        layer_stats = validation_results.get('layer_stats', {})
        
        # Filter hanya layer yang aktif
        active_layer_stats = {
            layer: count for layer, count in layer_stats.items()
            if layer in self.active_layers
        }
        
        if not active_layer_stats:
            return {
                'status': 'error',
                'message': "Tidak ada statistik layer aktif",
                'imbalance_score': 10.0
            }
            
        # Hitung statistik dasar
        total_objects = sum(active_layer_stats.values())
        if total_objects == 0:
            return {
                'status': 'error',
                'message': "Tidak ada objek layer aktif",
                'imbalance_score': 10.0
            }
            
        # Hitung persentase per layer
        layer_percentages = {
            layer: (count / total_objects) * 100
            for layer, count in active_layer_stats.items()
        }
        
        # Hitung ketidakseimbangan
        mean_percentage = 100 / len(active_layer_stats)
        max_deviation = max(abs(pct - mean_percentage) for pct in layer_percentages.values())
        
        # Normalisasi ketidakseimbangan ke skala 0-10
        imbalance_score = min(10.0, (max_deviation / mean_percentage) * 5)
        
        return {
            'status': 'success',
            'total_objects': total_objects,
            'layer_count': len(active_layer_stats),
            'layer_percentages': dict(sorted(layer_percentages.items(), key=lambda x: x[1], reverse=True)),
            'mean_percentage': mean_percentage,
            'max_deviation': max_deviation,
            'imbalance_score': imbalance_score
        }
    
    def analyze_bbox_statistics(
        self,
        split: str,
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis statistik bbox dalam dataset.
        
        Args:
            split: Split dataset
            sample_size: Jika > 0, gunakan sampel
            
        Returns:
            Dict statistik bounding box
        """
        images_dir = self.data_dir / split / 'images'
        labels_dir = self.data_dir / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            return {
                'status': 'error',
                'message': f"Direktori tidak lengkap",
                'bbox_count': 0
            }
        
        # Temukan semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        if not image_files:
            return {
                'status': 'error',
                'message': f"Tidak ada gambar ditemukan",
                'bbox_count': 0
            }
            
        # Jika sample_size ditentukan, ambil sampel acak
        if 0 < sample_size < len(image_files):
            random.seed(42)  # Untuk hasil yang konsisten
            image_files = random.sample(image_files, sample_size)
        
        # Analisis bbox
        width_vals = []
        height_vals = []
        area_vals = []
        aspect_ratio_vals = []
        class_bbox_sizes = {}
        
        for img_path in tqdm(image_files, desc="Analisis Bounding Box"):
            # Baca gambar untuk mendapatkan dimensi
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img_h, img_w = img.shape[:2]
                
                # Baca label
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                    
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        try:
                            cls_id = int(float(parts[0]))
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Skip koordinat tidak valid
                            if not all(0 <= x <= 1 for x in [x_center, y_center, width, height]):
                                continue
                                
                            # Convert ke pixel
                            pixel_width = width * img_w
                            pixel_height = height * img_h
                            pixel_area = pixel_width * pixel_height
                            
                            # Hitung rasio aspek
                            aspect_ratio = pixel_width / max(1, pixel_height)
                            
                            # Tambahkan ke statistik
                            width_vals.append(pixel_width)
                            height_vals.append(pixel_height)
                            area_vals.append(pixel_area)
                            aspect_ratio_vals.append(aspect_ratio)
                            
                            # Track per kelas
                            class_name = self.layer_config_manager.get_class_name(cls_id) or f"Unknown_{cls_id}"
                            if class_name not in class_bbox_sizes:
                                class_bbox_sizes[class_name] = {
                                    'widths': [],
                                    'heights': [],
                                    'areas': [],
                                    'count': 0
                                }
                                
                            class_bbox_sizes[class_name]['widths'].append(pixel_width)
                            class_bbox_sizes[class_name]['heights'].append(pixel_height)
                            class_bbox_sizes[class_name]['areas'].append(pixel_area)
                            class_bbox_sizes[class_name]['count'] += 1
                            
                        except (ValueError, IndexError, ZeroDivisionError):
                            continue
                            
            except Exception:
                continue
        
        # Hitung statistik
        if not width_vals:
            return {
                'status': 'error',
                'message': "Tidak ada bounding box valid ditemukan",
                'bbox_count': 0
            }
            
        # Statistik global
        bbox_stats = {
            'status': 'success',
            'bbox_count': len(width_vals),
            'mean_width': np.mean(width_vals),
            'mean_height': np.mean(height_vals),
            'mean_area': np.mean(area_vals),
            'mean_aspect_ratio': np.mean(aspect_ratio_vals),
            'min_width': np.min(width_vals),
            'max_width': np.max(width_vals),
            'min_height': np.min(height_vals),
            'max_height': np.max(height_vals),
            'min_area': np.min(area_vals),
            'max_area': np.max(area_vals)
        }
        
        # Statistik per kelas
        class_stats = {}
        for cls, stats in class_bbox_sizes.items():
            if stats['count'] > 0:
                class_stats[cls] = {
                    'count': stats['count'],
                    'mean_width': np.mean(stats['widths']),
                    'mean_height': np.mean(stats['heights']),
                    'mean_area': np.mean(stats['areas']),
                    'min_area': np.min(stats['areas']),
                    'max_area': np.max(stats['areas'])
                }
        
        bbox_stats['class_stats'] = class_stats
        
        return bbox_stats