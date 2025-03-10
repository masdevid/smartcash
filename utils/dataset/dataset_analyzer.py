"""
File: smartcash/utils/dataset/dataset_analyzer.py
Author: Alfrida Sabar
Deskripsi: Modul untuk analisis mendalam dataset dengan metrik statistik dan visualisasi (versi ringkas)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.dataset.dataset_utils import DatasetUtils

class DatasetAnalyzer:
    """Kelas untuk melakukan analisis mendalam tentang dataset."""
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        self.utils = DatasetUtils(config=config, data_dir=str(self.data_dir), logger=logger)
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict:
        """Analisis ukuran gambar dalam dataset."""
        split_path = self.utils.get_split_path(split)
        images_dir = split_path / 'images'
        
        if not images_dir.exists():
            return {'status': 'error', 'message': f"Direktori gambar tidak ditemukan: {images_dir}", 'sizes': {}}
        
        image_files = self.utils.find_image_files(images_dir, with_labels=False)
        if not image_files:
            return {'status': 'error', 'message': f"Tidak ada gambar ditemukan di {images_dir}", 'sizes': {}}
            
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
        
        # Analisis ukuran
        size_counts, aspect_ratios = {}, {}
        width_sum = height_sum = 0
        
        for img_path in tqdm(image_files, desc="Analisis Ukuran Gambar"):
            try:
                img = self.utils.load_image(img_path, target_size=None)
                if img is None or img.size == 0: continue
                    
                h, w = img.shape[:2]
                size_key = f"{w}x{h}"
                
                # Track ukuran dan rasio
                size_counts[size_key] = size_counts.get(size_key, 0) + 1
                aspect = round(w / h, 2)
                aspect_ratios[aspect] = aspect_ratios.get(aspect, 0) + 1
                
                width_sum += w
                height_sum += h
            except Exception:
                continue
        
        # Tentukan ukuran dan rasio dominan
        dominant_size = max(size_counts.items(), key=lambda x: x[1]) if size_counts else ("Unknown", 0)
        dominant_pct = (dominant_size[1] / len(image_files)) * 100 if size_counts else 0
        
        dominant_aspect = max(aspect_ratios.items(), key=lambda x: x[1]) if aspect_ratios else (0, 0)
        dominant_aspect_pct = (dominant_aspect[1] / len(image_files)) * 100 if aspect_ratios else 0
            
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
            'dominant_aspect_ratio_percent': dominant_aspect_pct,
            'mean_width': avg_width,
            'mean_height': avg_height
        }
    
    def analyze_class_balance(self, validation_results: Dict) -> Dict:
        """Analisis keseimbangan kelas dalam dataset."""
        class_stats = validation_results.get('class_stats', {})
        
        if not class_stats:
            return {'status': 'error', 'message': "Tidak ada statistik kelas", 'imbalance_score': 10.0}
            
        total_objects = sum(class_stats.values())
        if total_objects == 0:
            return {'status': 'error', 'message': "Tidak ada objek yang valid", 'imbalance_score': 10.0}
            
        # Hitung persentase dan ketidakseimbangan
        class_percentages = {cls: (count / total_objects) * 100 for cls, count in class_stats.items()}
        mean_percentage = 100 / len(class_stats)
        max_deviation = max(abs(pct - mean_percentage) for pct in class_percentages.values())
        
        # Skor ketidakseimbangan (0-10)
        imbalance_score = min(10.0, (max_deviation / mean_percentage) * 5)
        
        # Identifikasi kelas under/over-represented
        underrepresented = [cls for cls, pct in class_percentages.items() if pct < mean_percentage * 0.5]
        overrepresented = [cls for cls, pct in class_percentages.items() if pct > mean_percentage * 2]
        
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
        """Analisis keseimbangan layer dalam dataset."""
        layer_stats = validation_results.get('layer_stats', {})
        
        # Filter hanya layer yang aktif
        active_layer_stats = {layer: count for layer, count in layer_stats.items() if layer in self.active_layers}
        
        if not active_layer_stats:
            return {'status': 'error', 'message': "Tidak ada statistik layer aktif", 'imbalance_score': 10.0}
            
        total_objects = sum(active_layer_stats.values())
        if total_objects == 0:
            return {'status': 'error', 'message': "Tidak ada objek layer aktif", 'imbalance_score': 10.0}
            
        # Hitung ketidakseimbangan
        layer_percentages = {layer: (count / total_objects) * 100 for layer, count in active_layer_stats.items()}
        mean_percentage = 100 / len(active_layer_stats)
        max_deviation = max(abs(pct - mean_percentage) for pct in layer_percentages.values())
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
    
    def analyze_bbox_statistics(self, split: str, sample_size: int = 0) -> Dict:
        """Analisis statistik bbox dalam dataset."""
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
        
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            return {'status': 'error', 'message': f"Tidak ada gambar ditemukan di {images_dir}"}
            
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
        
        # Statistik bbox
        widths, heights, areas, aspect_ratios = [], [], [], []
        
        for img_path in tqdm(image_files, desc="Analisis BBox"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists(): continue
            
            for bbox_data in self.utils.parse_yolo_label(label_path):
                bbox = bbox_data.get('bbox')
                if not bbox: continue
                
                # Format YOLO: [x_center, y_center, width, height]
                width, height = bbox[2], bbox[3]
                widths.append(width)
                heights.append(height)
                areas.append(width * height)
                aspect_ratios.append(width / height if height > 0 else 0)
        
        if not widths:
            return {'status': 'error', 'message': "Tidak ada bbox valid ditemukan"}
        
        # Kategori ukuran bbox
        area_categories = {
            'small': sum(1 for a in areas if a < 0.02),    # < 2%
            'medium': sum(1 for a in areas if 0.02 <= a <= 0.1),  # 2-10%
            'large': sum(1 for a in areas if a > 0.1),     # > 10%
            'total': len(areas)
        }
        
        # Persentase
        for size in ['small', 'medium', 'large']:
            area_categories[f'{size}_pct'] = (area_categories[size] / len(areas)) * 100
            
        return {
            'status': 'success',
            'total_bbox': len(widths),
            'width': {'min': min(widths), 'max': max(widths), 'mean': np.mean(widths)},
            'height': {'min': min(heights), 'max': max(heights), 'mean': np.mean(heights)},
            'area': {'min': min(areas), 'max': max(areas), 'mean': np.mean(areas)},
            'aspect_ratio': {'min': min(aspect_ratios), 'max': max(aspect_ratios), 'mean': np.mean(aspect_ratios)},
            'area_categories': area_categories
        }