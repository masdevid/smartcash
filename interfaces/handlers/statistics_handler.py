# File: src/metrics/statistics_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk analisis statistik dataset dengan penanganan error yang kuat

import numpy as np
from typing import Dict, Any, Union
from pathlib import Path

from config.manager import ConfigManager
from interfaces.handlers.base_handler import BaseHandler
from utils.logging import ColoredLogger

class DatasetStatistics:
    """Kelas untuk menyimpan dan mengelola statistik dataset"""
    def __init__(self):
        # Basic stats
        self.total_images = 0
        self.total_labels = 0
        self.split_distribution = {}
        
        # Class distribution
        self.class_counts = {}
        self.class_per_split = {}
        
        # Image stats
        self.image_sizes = []
        self.aspect_ratios = []
        self.file_sizes = []
        
        # Label stats
        self.boxes_per_image = []
        self.box_sizes = []
        self.box_aspects = []
        
        # Quality metrics
        self.blur_scores = []
        self.brightness_scores = []
        self.contrast_scores = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Konversi statistik ke dictionary untuk kemudahan akses"""
        return {
            'total_images': self.total_images,
            'total_labels': self.total_labels,
            'split_distribution': self.split_distribution,
            'class_counts': self.class_counts,
            'class_per_split': self.class_per_split,
            'image_sizes': self.image_sizes,
            'aspect_ratios': self.aspect_ratios,
            'file_sizes': self.file_sizes,
            'boxes_per_image': self.boxes_per_image,
            'box_sizes': self.box_sizes,
            'box_aspects': self.box_aspects,
            'blur_scores': self.blur_scores,
            'brightness_scores': self.brightness_scores,
            'contrast_scores': self.contrast_scores
        }

class DataStatisticsHandler(BaseHandler):
    """Handler untuk analisis statistik dataset dengan penanganan error yang kuat"""
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.logger = ColoredLogger('DataStatistics')
        
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analisis komprehensif dataset dengan penanganan error yang robust
        
        Returns:
            Dict: Statistik dan analisis dataset
        """
        try:
            # Inisialisasi objek statistik
            stats = DatasetStatistics()
            
            # Analisis setiap split
            for split in ['train', 'val', 'test']:
                self._analyze_split(split, stats)
            
            # Analisis lanjutan
            return {
                'stats': stats.to_dict(),
                'augmentation_recommendations': self.get_augmentation_recommendations(stats),
                'class_balance': self.analyze_class_balance(stats),
                'quality_issues': self.analyze_quality_issues(stats)
            }
        except Exception as e:
            self.logger.error(f"Kesalahan dalam analisis dataset: {str(e)}")
            return {}
    
    def _analyze_split(self, split: str, stats: DatasetStatistics):
        """
        Analisis statistik untuk satu split dataset
        
        Args:
            split (str): Nama split (train/val/test)
            stats (DatasetStatistics): Objek untuk menyimpan statistik
        """
        img_dir = self.rupiah_dir / split / 'images'
        label_dir = self.rupiah_dir / split / 'labels'
        
        if not img_dir.exists() or not label_dir.exists():
            return
        
        split_stats = {
            'images': 0,
            'labels': 0,
            'class_dist': {}
        }
        
        # Proses setiap gambar
        for img_path in img_dir.glob('*.jpg'):
            # Hitung statistik gambar
            split_stats['images'] += 1
            stats.total_images += 1
            
            # Analisis ukuran gambar
            try:
                img = self._analyze_image(img_path)
                if img:
                    stats.image_sizes.append(img['size'])
                    stats.aspect_ratios.append(img['aspect'])
                    stats.file_sizes.append(img['file_size'])
                    stats.blur_scores.append(img['blur_score'])
                    stats.brightness_scores.append(img['brightness'])
                    stats.contrast_scores.append(img['contrast'])
            except Exception as e:
                self.logger.error(f"Gagal menganalisis gambar {img_path}: {str(e)}")
            
            # Analisis label
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                try:
                    label_stats = self._analyze_label(label_path)
                    split_stats['labels'] += 1
                    stats.total_labels += 1
                    
                    # Distribusi kelas
                    for cls, count in label_stats['class_dist'].items():
                        # Update distribusi kelas
                        split_stats['class_dist'][cls] = split_stats['class_dist'].get(cls, 0) + count
                        stats.class_counts[cls] = stats.class_counts.get(cls, 0) + count
                        
                        # Update distribusi kelas per split
                        if split not in stats.class_per_split:
                            stats.class_per_split[split] = {}
                        stats.class_per_split[split][cls] = stats.class_per_split[split].get(cls, 0) + count
                    
                    # Statistik kotak
                    stats.boxes_per_image.append(label_stats['num_boxes'])
                    stats.box_sizes.extend(label_stats['box_sizes'])
                    stats.box_aspects.extend(label_stats['box_aspects'])
                except Exception as e:
                    self.logger.error(f"Gagal menganalisis label {label_path}: {str(e)}")
        
        # Simpan statistik split
        stats.split_distribution[split] = split_stats
    
    def _analyze_image(self, img_path: Path) -> Dict:
        """
        Analisis statistik untuk satu gambar
        
        Args:
            img_path (Path): Path ke file gambar
        
        Returns:
            Dict: Statistik gambar
        """
        import cv2
        
        # Baca gambar
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Hitung statistik
        h, w = img.shape[:2]
        file_size = img_path.stat().st_size / 1024  # KB
        
        # Metrik kualitas
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        return {
            'size': (w, h),
            'aspect': w/h if h != 0 else 1,
            'file_size': file_size,
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast
        }
    
    def _analyze_label(self, label_path: Path) -> Dict:
        """
        Analisis statistik untuk satu file label
        
        Args:
            label_path (Path): Path ke file label
        
        Returns:
            Dict: Statistik label
        """
        stats = {
            'num_boxes': 0,
            'class_dist': {},
            'box_sizes': [],
            'box_aspects': []
        }
        
        with open(label_path) as f:
            for line in f:
                try:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:  # class, x, y, w, h
                        stats['num_boxes'] += 1
                        
                        # Distribusi kelas
                        cls = int(values[0])
                        stats['class_dist'][cls] = stats['class_dist'].get(cls, 0) + 1
                        
                        # Statistik kotak
                        stats['box_sizes'].append(values[3] * values[4])  # area
                        stats['box_aspects'].append(values[3] / max(values[4], 1e-6))  # w/h rasio
                except Exception as e:
                    self.logger.error(f"Kesalahan parsing label: {str(e)}")
        
        return stats
    
    def analyze_class_balance(self, dataset_stats: Union[DatasetStatistics, Dict]) -> Dict:
        """
        Analisis keseimbangan kelas
        
        Args:
            dataset_stats (Union[DatasetStatistics, Dict]): Statistik dataset
        
        Returns:
            Dict: Analisis keseimbangan kelas
        """
        analysis = {
            'class_ratios': {},
            'imbalance_issues': [],
            'recommendations': []
        }
        
        try:
            # Konversi ke dictionary jika diperlukan
            if hasattr(dataset_stats, 'to_dict'):
                stats_dict = dataset_stats.to_dict()
            elif isinstance(dataset_stats, dict):
                stats_dict = dataset_stats.get('stats', {})
            else:
                return analysis
            
            # Ambil class_counts
            class_counts = stats_dict.get('class_counts', {})
            
            # Hindari pembagian dengan nol
            total_objects = sum(class_counts.values())
            if total_objects == 0:
                return analysis
            
            # Rasio ideal
            ideal_ratio = 1.0 / len(class_counts)
            
            # Analisis setiap kelas
            for cls, count in class_counts.items():
                ratio = count / total_objects
                analysis['class_ratios'][cls] = ratio
                
                # Identifikasi ketidakseimbangan
                if ratio < ideal_ratio * 0.5:
                    analysis['imbalance_issues'].append({
                        'class': cls,
                        'issue': 'Underrepresented',
                        'current_ratio': ratio,
                        'ideal_ratio': ideal_ratio
                    })
                elif ratio > ideal_ratio * 1.5:
                    analysis['imbalance_issues'].append({
                        'class': cls,
                        'issue': 'Overrepresented',
                        'current_ratio': ratio,
                        'ideal_ratio': ideal_ratio
                    })
            
            return analysis
        except Exception as e:
            self.logger.error(f"Kesalahan dalam analisis keseimbangan kelas: {str(e)}")
            return analysis
    
    
    def analyze_quality_issues(self, dataset_stats: Union[DatasetStatistics, Dict]) -> Dict:
        """
        Analisis masalah kualitas dataset
        
        Args:
            dataset_stats (Union[DatasetStatistics, Dict]): Objek statistik dataset
        
        Returns:
            Dict: Analisis masalah kualitas
        """
        analysis = {
            'blur_issues': [],
            'lighting_issues': [],
            'recommendations': []
        }
        
        try:
            # Konversi ke dictionary jika diperlukan
            if hasattr(dataset_stats, 'to_dict'):
                stats_dict = dataset_stats.to_dict()
            elif isinstance(dataset_stats, dict):
                stats_dict = dataset_stats.get('stats', {})
            else:
                return analysis
            
            # Ambil skor-skor
            blur_scores = stats_dict.get('blur_scores', [])
            brightness_scores = stats_dict.get('brightness_scores', [])
            
            # Hindari kesalahan dengan pemeriksaan panjang
            if not blur_scores or not brightness_scores:
                return analysis
            
            # Analisis blur
            blur_mean = np.mean(blur_scores)
            blur_std = np.std(blur_scores)
            blur_threshold = blur_mean - 1 * blur_std
            
            for score in blur_scores:
                if score < blur_threshold:
                    analysis['blur_issues'].append({
                        'score': score,
                        'threshold': blur_threshold
                    })
            
            # Analisis pencahayaan
            brightness_mean = np.mean(brightness_scores)
            brightness_std = np.std(brightness_scores)
            
            for brightness in brightness_scores:
                if brightness < brightness_mean - 2*brightness_std:
                    analysis['lighting_issues'].append({
                        'type': 'Too Dark',
                        'value': brightness,
                        'threshold': brightness_mean - 2*brightness_std
                    })
                elif brightness > brightness_mean + 2*brightness_std:
                    analysis['lighting_issues'].append({
                        'type': 'Too Bright',
                        'value': brightness,
                        'threshold': brightness_mean + 2*brightness_std
                    })
            
            # Rekomendasi
            if analysis['blur_issues']:
                analysis['recommendations'].append({
                    'issue': 'Blur',
                    'action': 'Pertimbangkan filter atau re-capture gambar dengan kualitas lebih baik',
                    'affected_count': len(analysis['blur_issues'])
                })
            
            if analysis['lighting_issues']:
                analysis['recommendations'].append({
                    'issue': 'Pencahayaan',
                    'action': 'Normalisasi pencahayaan atau tambahkan augmentasi pencahayaan',
                    'affected_count': len(analysis['lighting_issues'])
                })
            
            return analysis
        except Exception as e:
            self.logger.error(f"Kesalahan dalam analisis kualitas: {str(e)}")
            return analysis
    
    def get_augmentation_recommendations(self, dataset_stats: Union[DatasetStatistics, Dict]) -> Dict:
        """
        Hasilkan rekomendasi augmentasi
        
        Args:
            dataset_stats (Union[DatasetStatistics, Dict]): Objek statistik dataset
        
        Returns:
            Dict: Rekomendasi augmentasi
        """
        recommendations = {
            'general': [],
            'per_class': {}
        }
        
        try:
            # Konversi ke dictionary jika diperlukan
            if hasattr(dataset_stats, 'to_dict'):
                stats_dict = dataset_stats.to_dict()
            elif isinstance(dataset_stats, dict):
                stats_dict = dataset_stats.get('stats', {})
            else:
                return recommendations
            
            # Ambil class_counts
            class_counts = stats_dict.get('class_counts', {})
            
            # Hindari kesalahan pembagian dengan nol
            if not class_counts:
                return recommendations
            
            # Hitung total objek
            total_objects = sum(class_counts.values())
            ideal_ratio = 1.0 / len(class_counts)
            
            # Rekomendasi umum
            recommendations['general'].append({
                'type': 'Class Balance',
                'description': 'Periksa distribusi kelas dataset',
                'actions': [
                    'Pertimbangkan augmentasi untuk kelas dengan representasi rendah',
                    'Seimbangkan distribusi kelas'
                ]
            })
            
            # Rekomendasi per kelas
            for cls, count in class_counts.items():
                ratio = count / total_objects
                class_recs = []
                
                if ratio < ideal_ratio * 0.5:
                    class_recs.append({
                        'priority': 'High',
                        'action': f'Tambahkan augmentasi untuk kelas {cls}'
                    })
                elif ratio < ideal_ratio:
                    class_recs.append({
                        'priority': 'Medium',
                        'action': f'Pertimbangkan augmentasi ringan untuk kelas {cls}'
                    })
                
                if class_recs:
                    recommendations['per_class'][cls] = class_recs
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Kesalahan dalam menghasilkan rekomendasi augmentasi: {str(e)}")
            return recommendations