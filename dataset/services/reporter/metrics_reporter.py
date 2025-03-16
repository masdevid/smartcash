"""
File: smartcash/dataset/services/reporter/metrics_reporter.py
Deskripsi: Komponen untuk menghitung dan melaporkan metrik-metrik dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config


class MetricsReporter:
    """Komponen untuk menghitung dan melaporkan metrik-metrik dataset."""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi MetricsReporter.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("metrics_reporter")
        
        # Setup layer config
        self.layer_config = get_layer_config()
        
        self.logger.info(f"📊 MetricsReporter diinisialisasi untuk perhitungan dan analisis metrik dataset")
    
    def calculate_class_metrics(self, class_stats: Dict[str, int]) -> Dict[str, Any]:
        """
        Hitung metrik-metrik terkait distribusi kelas.
        
        Args:
            class_stats: Dictionary berisi jumlah per kelas
            
        Returns:
            Dictionary berisi metrik-metrik kelas
        """
        metrics = {
            'total_objects': sum(class_stats.values()),
            'num_classes': len(class_stats),
            'most_common': None,
            'least_common': None,
            'evenness': 0,
            'imbalance_score': 0,
            'class_percentages': {}
        }
        
        if not class_stats:
            return metrics
            
        # Hitung persentase per kelas
        for cls, count in class_stats.items():
            metrics['class_percentages'][cls] = (count / metrics['total_objects']) * 100
            
        # Identifikasi kelas paling umum dan paling jarang
        sorted_stats = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        metrics['most_common'] = {'class': sorted_stats[0][0], 'count': sorted_stats[0][1]}
        metrics['least_common'] = {'class': sorted_stats[-1][0], 'count': sorted_stats[-1][1]}
        
        # Hitung metrik evenness (Pielou's evenness)
        counts = np.array(list(class_stats.values()))
        if metrics['num_classes'] > 1 and metrics['total_objects'] > 0:
            # Shannon diversity dan max diversity
            proportions = counts / metrics['total_objects']
            shannon_div = -np.sum(proportions * np.log(proportions + 1e-10))
            max_div = np.log(metrics['num_classes'])
            metrics['evenness'] = shannon_div / max_div if max_div > 0 else 0
            
            # Imbalance score (0-10 scale, 0 = seimbang, 10 = sangat tidak seimbang)
            imb_ratio = metrics['most_common']['count'] / max(1, metrics['least_common']['count'])
            metrics['imbalance_score'] = min(10.0, ((imb_ratio - 1) * 10) / 99)
            
        return metrics
    
    def calculate_layer_metrics(self, layer_stats: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Hitung metrik-metrik terkait distribusi layer.
        
        Args:
            layer_stats: Dictionary berisi jumlah per layer
            
        Returns:
            Dictionary berisi metrik-metrik layer
        """
        total_layer_counts = {layer: sum(counts.values()) for layer, counts in layer_stats.items()}
        
        metrics = {
            'total_objects': sum(total_layer_counts.values()),
            'num_layers': len(layer_stats),
            'most_common': None,
            'least_common': None,
            'evenness': 0,
            'imbalance_score': 0,
            'layer_percentages': {}
        }
        
        if not layer_stats:
            return metrics
            
        # Hitung persentase per layer
        for layer, count in total_layer_counts.items():
            metrics['layer_percentages'][layer] = (count / metrics['total_objects']) * 100
            
        # Identifikasi layer paling umum dan paling jarang
        sorted_stats = sorted(total_layer_counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_stats:
            metrics['most_common'] = {'layer': sorted_stats[0][0], 'count': sorted_stats[0][1]}
            metrics['least_common'] = {'layer': sorted_stats[-1][0], 'count': sorted_stats[-1][1]}
        
        # Hitung metrics evenness dan imbalance
        if metrics['num_layers'] > 1 and metrics['total_objects'] > 0:
            counts = np.array(list(total_layer_counts.values()))
            proportions = counts / metrics['total_objects']
            shannon_div = -np.sum(proportions * np.log(proportions + 1e-10))
            max_div = np.log(metrics['num_layers'])
            metrics['evenness'] = shannon_div / max_div if max_div > 0 else 0
            
            # Imbalance score (0-10 scale)
            if metrics['most_common'] and metrics['least_common'] and metrics['least_common']['count'] > 0:
                imb_ratio = metrics['most_common']['count'] / metrics['least_common']['count']
                metrics['imbalance_score'] = min(10.0, ((imb_ratio - 1) * 10) / 99)
            
        return metrics
    
    def calculate_bbox_metrics(self, bbox_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hitung metrik-metrik terkait bounding box.
        
        Args:
            bbox_stats: Statistik bounding box
            
        Returns:
            Dictionary berisi metrik-metrik bbox yang diperkaya
        """
        # Dapatkan informasi dasar
        total_bbox = bbox_stats.get('total_bbox', 0)
        
        metrics = {
            'total_bbox': total_bbox,
            'size_distribution': {},
            'aspect_ratio_distribution': {},
            'size_diversity': 0,
            'size_uniformity': 0
        }
        
        if total_bbox == 0:
            return metrics
            
        # Kategori ukuran dari statistik area
        area_cats = bbox_stats.get('area_categories', {})
        metrics['size_distribution'] = {
            'small': area_cats.get('small', 0),
            'small_pct': area_cats.get('small_pct', 0),
            'medium': area_cats.get('medium', 0),
            'medium_pct': area_cats.get('medium_pct', 0),
            'large': area_cats.get('large', 0),
            'large_pct': area_cats.get('large_pct', 0)
        }
        
        # Kategorisasi aspect ratio
        if 'aspect_ratio' in bbox_stats:
            ar_stats = bbox_stats['aspect_ratio']
            ar_mean = ar_stats.get('mean', 1.0)
            ar_std = ar_stats.get('std', 0.0)
            
            metrics['aspect_ratio_distribution'] = {
                'portrait': 0,    # < 0.8
                'square': 0,      # 0.8 - 1.2
                'landscape': 0,   # > 1.2
                'extreme': 0      # < 0.5 atau > 2.0
            }
            
            # Estimasi distribusi (karena kita tidak punya histogram asli)
            # Asumsi distribusi normal
            width = bbox_stats.get('width', {})
            height = bbox_stats.get('height', {})
            
            if width and height and 'mean' in width and 'mean' in height:
                w_mean, h_mean = width['mean'], height['mean']
                avg_ar = w_mean / h_mean if h_mean > 0 else 1.0
                
                # Estimasi persentase distribusi aspect ratio
                metrics['aspect_ratio_distribution']['portrait'] = 35 if avg_ar < 0.9 else 20
                metrics['aspect_ratio_distribution']['square'] = 40 if 0.9 <= avg_ar <= 1.1 else 30
                metrics['aspect_ratio_distribution']['landscape'] = 35 if avg_ar > 1.1 else 20
                
                # Estimasi persentase extreme aspect ratio
                metrics['aspect_ratio_distribution']['extreme'] = 5 if ar_std > 0.5 else 2
            
            # Size diversity & uniformity
            size_diversity = ar_std / ar_mean if ar_mean > 0 else 0
            metrics['size_diversity'] = min(1.0, size_diversity * 2)  # Scale to 0-1
            metrics['size_uniformity'] = max(0, 1.0 - metrics['size_diversity'])
        
        return metrics
    
    def calculate_dataset_quality_score(self, metrics: Dict[str, Any]) -> float:
        """
        Hitung skor kualitas dataset berdasarkan metrik-metrik yang ada.
        
        Args:
            metrics: Metrik-metrik dataset
            
        Returns:
            Skor kualitas dataset (0-100)
        """
        score = 50.0  # Nilai default
        factors = []
        
        # Faktor keseimbangan kelas (30%)
        class_metrics = metrics.get('class_metrics', {})
        if class_metrics:
            class_imbalance = class_metrics.get('imbalance_score', 0)
            class_evenness = class_metrics.get('evenness', 0)
            
            # 10 - imbalance_score (0 terbaik, 10 terburuk) / 10 * 30
            balance_factor = (10 - class_imbalance) / 10 * 30
            factors.append(('Keseimbangan Kelas', balance_factor))
        
        # Faktor kecukupan data (20%)
        total_samples = 0
        for split in ['train', 'valid', 'test']:
            split_stats = metrics.get('split_stats', {}).get(split, {})
            total_samples += split_stats.get('images', 0)
        
        # Aturan sederhana: 
        # - <500 sampel: 5 poin
        # - 500-1000 sampel: 10 poin
        # - 1000-5000 sampel: 15 poin
        # - >5000 sampel: 20 poin
        data_factor = 5
        if total_samples >= 500: data_factor = 10
        if total_samples >= 1000: data_factor = 15
        if total_samples >= 5000: data_factor = 20
        factors.append(('Kecukupan Data', data_factor))
        
        # Faktor keseimbangan ukuran bbox (15%)
        bbox_metrics = metrics.get('bbox_metrics', {})
        if bbox_metrics:
            size_dist = bbox_metrics.get('size_distribution', {})
            small_pct = size_dist.get('small_pct', 0)
            large_pct = size_dist.get('large_pct', 0)
            
            # Penalti untuk distribusi ukuran yang sangat tidak seimbang
            size_balance = 15
            if small_pct > 80 or large_pct > 80:
                size_balance = 5
            elif small_pct > 60 or large_pct > 60:
                size_balance = 10
                
            factors.append(('Keseimbangan Ukuran Bbox', size_balance))
        
        # Faktor keseimbangan split (15%)
        split_imbalance = 0
        split_counts = {}
        for split in ['train', 'valid', 'test']:
            split_stats = metrics.get('split_stats', {}).get(split, {})
            split_counts[split] = split_stats.get('images', 0)
            
        if split_counts.get('train', 0) > 0:
            # Hitung rasio ideal (70-15-15)
            total = sum(split_counts.values())
            train_ratio = split_counts.get('train', 0) / total if total > 0 else 0
            valid_ratio = split_counts.get('valid', 0) / total if total > 0 else 0
            test_ratio = split_counts.get('test', 0) / total if total > 0 else 0
            
            # Perbedaan dari rasio ideal
            train_diff = abs(train_ratio - 0.7)
            valid_diff = abs(valid_ratio - 0.15)
            test_diff = abs(test_ratio - 0.15)
            
            # Total perbedaan (0 terbaik, 2 terburuk)
            total_diff = train_diff + valid_diff + test_diff
            split_factor = (1 - total_diff/2) * 15
            
            factors.append(('Keseimbangan Split', max(0, split_factor)))
            
        # Faktor kualitas anotasi (sisanya, 20%)
        # Kita bisa mengestimasi dari persentase gambar yang memiliki label
        label_ratio = 0
        for split in ['train', 'valid', 'test']:
            split_stats = metrics.get('split_stats', {}).get(split, {})
            images = split_stats.get('images', 0)
            labels = split_stats.get('labels', 0)
            
            if images > 0:
                label_ratio += labels / images
                
        # Rata-rata rasio label
        if len(metrics.get('split_stats', {})) > 0:
            label_ratio /= len(metrics.get('split_stats', {}))
            annotation_factor = label_ratio * 20
            factors.append(('Kualitas Anotasi', annotation_factor))
        
        # Hitung total skor
        score = sum(factor[1] for factor in factors)
        
        return min(100, max(0, score))
    
    def generate_metrics_report(self, 
                             class_stats: Dict[str, int], 
                             layer_stats: Dict[str, Dict[str, int]],
                             bbox_stats: Dict[str, Any],
                             split_stats: Dict[str, Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Generate laporan metrik yang komprehensif.
        
        Args:
            class_stats: Statistik kelas
            layer_stats: Statistik layer
            bbox_stats: Statistik bounding box
            split_stats: Statistik split dataset (opsional)
            
        Returns:
            Dictionary berisi laporan metrik
        """
        report = {
            'class_metrics': self.calculate_class_metrics(class_stats),
            'layer_metrics': self.calculate_layer_metrics(layer_stats),
            'bbox_metrics': self.calculate_bbox_metrics(bbox_stats),
            'split_stats': split_stats or {}
        }
        
        # Hitung quality score
        report['quality_score'] = self.calculate_dataset_quality_score(report)
        
        # Buat rekomendasi
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate rekomendasi berdasarkan metrik-metrik dataset.
        
        Args:
            metrics: Metrik-metrik dataset
            
        Returns:
            List rekomendasi
        """
        recommendations = []
        
        # Rekomendasi berdasarkan imbalance kelas
        class_metrics = metrics.get('class_metrics', {})
        imbalance_score = class_metrics.get('imbalance_score', 0)
        
        if imbalance_score > 7:
            recommendations.append("🚨 Dataset sangat tidak seimbang. Gunakan teknik balancing seperti oversampling kelas minoritas atau undersampling kelas mayoritas.")
        elif imbalance_score > 5:
            recommendations.append("⚠️ Dataset cukup tidak seimbang. Pertimbangkan menambah sampel pada kelas minoritas melalui augmentasi.")
            
        # Rekomendasi berdasarkan ukuran dataset
        total_samples = 0
        for split in ['train', 'valid', 'test']:
            split_stats = metrics.get('split_stats', {}).get(split, {})
            total_samples += split_stats.get('images', 0)
            
        if total_samples < 500:
            recommendations.append("📉 Dataset terlalu kecil. Tambahkan lebih banyak data training untuk meningkatkan performa model.")
        elif total_samples < 1000:
            recommendations.append("📊 Ukuran dataset cukup, tapi lebih banyak data akan membantu meningkatkan performa model.")
            
        # Rekomendasi berdasarkan distribusi ukuran bbox
        bbox_metrics = metrics.get('bbox_metrics', {})
        size_dist = bbox_metrics.get('size_distribution', {})
        
        if size_dist.get('small_pct', 0) > 70:
            recommendations.append("🔍 Terlalu banyak bounding box kecil. Model mungkin kesulitan mendeteksi objek kecil. Pertimbangkan augmentasi khusus untuk objek kecil.")
        
        if size_dist.get('large_pct', 0) < 10 and size_dist.get('medium_pct', 0) < 30:
            recommendations.append("📏 Terlalu sedikit bounding box besar/sedang. Tambahkan sampel dengan objek yang lebih besar untuk keseimbangan.")
            
        # Rekomendasi berdasarkan rasio split
        split_counts = {}
        for split in ['train', 'valid', 'test']:
            split_stats = metrics.get('split_stats', {}).get(split, {})
            split_counts[split] = split_stats.get('images', 0)
            
        total_split = sum(split_counts.values())
        if total_split > 0:
            train_ratio = split_counts.get('train', 0) / total_split
            valid_ratio = split_counts.get('valid', 0) / total_split
            test_ratio = split_counts.get('test', 0) / total_split
            
            if train_ratio < 0.6:
                recommendations.append("📚 Training set terlalu kecil relatif terhadap total dataset. Idealnya 70-80% data untuk training.")
            if valid_ratio < 0.1:
                recommendations.append("📋 Validation set terlalu kecil. Idealnya sekitar 15% dari total dataset.")
            if test_ratio < 0.1:
                recommendations.append("📝 Test set terlalu kecil. Idealnya sekitar 15% dari total dataset.")
                
        # Rekomendasi umum jika skor kualitas rendah
        quality_score = metrics.get('quality_score', 0)
        if quality_score < 50:
            recommendations.append("🔄 Dataset memiliki kualitas rendah. Fokus pada peningkatan jumlah dan keseimbangan data.")
        elif quality_score < 70:
            recommendations.append("🔍 Dataset memiliki kualitas sedang. Periksa rekomendasi spesifik untuk meningkatkan kualitas data.")
            
        return recommendations