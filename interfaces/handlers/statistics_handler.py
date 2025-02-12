# File: src/interfaces/handlers/statistics_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk analisis statistik dataset

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from interfaces.handlers.base_handler import BaseHandler

class DatasetStatistics:
    """Data class untuk menyimpan statistik dataset"""
    def __init__(self):
        # Basic stats
        self.total_images = 0
        self.total_labels = 0
        self.split_distribution = {}
        
        # Class distribution
        self.class_counts = defaultdict(int)
        self.class_per_split = defaultdict(lambda: defaultdict(int))
        
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

class DataStatisticsHandler(BaseHandler):
    """Handler untuk analisis statistik dataset"""
    def __init__(self, config):
        super().__init__(config)
        self.stats = DatasetStatistics()
        
        # Ensure all necessary directories exist
        try:
            # Create rupiah directory if it doesn't exist
            self.rupiah_dir.mkdir(parents=True, exist_ok=True)
            
            # Create stats directory with full path
            self.output_dir = self.rupiah_dir / 'stats'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat direktori stats: {str(e)}")
        
    def analyze_dataset(self) -> DatasetStatistics:
        """Analisis komprehensif dataset"""
        self.logger.info("📊 Memulai analisis statistik dataset...")
        
        try:
            # Reset statistics
            self.stats = DatasetStatistics()
            
            # Analyze each split
            for split in ['train', 'val', 'test']:
                split_stats = self._analyze_split(split)
                self.stats.split_distribution[split] = split_stats
                
            # Generate visualizations
            self._generate_visualizations()
            
            # Update operation stats
            self.update_stats('analysis', self.stats.__dict__)
            self.log_operation("Analisis dataset", "success")
            
        except Exception as e:
            self.log_operation("Analisis dataset", "failed", str(e))
            
        return self.stats
        
    def _analyze_split(self, split: str) -> Dict:
        """Analisis statistik untuk satu split"""
        img_dir = self.rupiah_dir / split / 'images'
        label_dir = self.rupiah_dir / split / 'labels'
        
        if not img_dir.exists() or not label_dir.exists():
            return {
                'images': 0,
                'labels': 0,
                'class_dist': {},
                'avg_boxes': 0
            }
            
        split_stats = {
            'images': 0,
            'labels': 0,
            'class_dist': defaultdict(int),
            'avg_boxes': 0
        }
        
        # Process each image
        total_boxes = 0
        for img_path in img_dir.glob('*.jpg'):
            # Image statistics
            img_stats = self._analyze_image(img_path)
            if img_stats:
                split_stats['images'] += 1
                self.stats.image_sizes.append(img_stats['size'])
                self.stats.aspect_ratios.append(img_stats['aspect'])
                self.stats.file_sizes.append(img_stats['file_size'])
                self.stats.blur_scores.append(img_stats['blur_score'])
                self.stats.brightness_scores.append(img_stats['brightness'])
                self.stats.contrast_scores.append(img_stats['contrast'])
            
            # Label statistics
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label_stats = self._analyze_label(label_path)
                split_stats['labels'] += 1
                total_boxes += label_stats['num_boxes']
                
                # Update class distribution
                for cls, count in label_stats['class_dist'].items():
                    split_stats['class_dist'][cls] += count
                    self.stats.class_counts[cls] += count
                    self.stats.class_per_split[split][cls] += count
                
                # Update box statistics
                self.stats.boxes_per_image.append(label_stats['num_boxes'])
                self.stats.box_sizes.extend(label_stats['box_sizes'])
                self.stats.box_aspects.extend(label_stats['box_aspects'])
        
        # Calculate averages
        if split_stats['images'] > 0:
            split_stats['avg_boxes'] = total_boxes / split_stats['images']
            
        return split_stats
        
    def _analyze_image(self, img_path: Path) -> Optional[Dict]:
        """Analisis statistik untuk satu gambar"""
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                return None
                
            # Basic stats
            h, w = img.shape[:2]
            file_size = img_path.stat().st_size / 1024  # KB
            
            # Quality metrics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            return {
                'size': (w, h),
                'aspect': w/h,
                'file_size': file_size,
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {img_path.name}: {str(e)}")
            return None
            
    def _analyze_label(self, label_path: Path) -> Dict:
        """Analisis statistik untuk satu file label"""
        stats = {
            'num_boxes': 0,
            'class_dist': defaultdict(int),
            'box_sizes': [],
            'box_aspects': []
        }
        
        try:
            with open(label_path) as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:  # class, x, y, w, h
                        stats['num_boxes'] += 1
                        stats['class_dist'][int(values[0])] += 1
                        
                        # Box statistics (normalized)
                        stats['box_sizes'].append(values[3] * values[4])  # area
                        stats['box_aspects'].append(values[3] / values[4])  # w/h
                        
        except Exception as e:
            self.logger.error(f"Error analyzing {label_path.name}: {str(e)}")
            
        return stats
        
    def _generate_visualizations(self):
        """Generate visualization plots"""
        try:
            # Plot settings
            plt.style.use('seaborn')
            
            # 1. Class Distribution
            self._plot_class_distribution()
            
            # 2. Split Distribution
            self._plot_split_distribution()
            
            # 3. Image Statistics
            self._plot_image_statistics()
            
            # 4. Box Statistics
            self._plot_box_statistics()
            
            # 5. Quality Metrics
            self._plot_quality_metrics()
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            
    def _plot_class_distribution(self):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        
        # Overall distribution
        plt.subplot(1, 2, 1)
        classes = list(self.stats.class_counts.keys())
        counts = list(self.stats.class_counts.values())
        plt.bar(classes, counts)
        plt.title('Overall Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Per-split distribution
        plt.subplot(1, 2, 2)
        splits = list(self.stats.class_per_split.keys())
        data = []
        for split in splits:
            data.append(list(self.stats.class_per_split[split].values()))
        plt.bar(classes, data[0], label=splits[0])
        for i in range(1, len(splits)):
            plt.bar(classes, data[i], bottom=sum(data[:i]), label=splits[i])
        plt.title('Class Distribution per Split')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close()
        
    def _plot_split_distribution(self):
        """Plot split distribution"""
        plt.figure(figsize=(8, 6))
        
        splits = list(self.stats.split_distribution.keys())
        images = [s['images'] for s in self.stats.split_distribution.values()]
        labels = [s['labels'] for s in self.stats.split_distribution.values()]
        
        x = np.arange(len(splits))
        width = 0.35
        
        plt.bar(x - width/2, images, width, label='Images')
        plt.bar(x + width/2, labels, width, label='Labels')
        
        plt.xlabel('Split')
        plt.ylabel('Count')
        plt.title('Dataset Split Distribution')
        plt.xticks(x, splits)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'split_distribution.png')
        plt.close()
        
    def _plot_image_statistics(self):
        """Plot image statistics"""
        plt.figure(figsize=(15, 5))
        
        # Image sizes
        plt.subplot(1, 3, 1)
        sizes = np.array(self.stats.image_sizes)
        plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.5)
        plt.title('Image Dimensions')
        plt.xlabel('Width')
        plt.ylabel('Height')
        
        # Aspect ratios
        plt.subplot(1, 3, 2)
        plt.hist(self.stats.aspect_ratios, bins=30)
        plt.title('Aspect Ratio Distribution')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Count')
        
        # File sizes
        plt.subplot(1, 3, 3)
        plt.hist(self.stats.file_sizes, bins=30)
        plt.title('File Size Distribution')
        plt.xlabel('Size (KB)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_statistics.png')
        plt.close()
        
    def _plot_box_statistics(self):
        """Plot bounding box statistics"""
        plt.figure(figsize=(15, 5))
        
        # Boxes per image
        plt.subplot(1, 3, 1)
        plt.hist(self.stats.boxes_per_image, bins=max(10, max(self.stats.boxes_per_image)))
        plt.title('Objects per Image')
        plt.xlabel('Number of Objects')
        plt.ylabel('Count')
        
        # Box sizes
        plt.subplot(1, 3, 2)
        plt.hist(self.stats.box_sizes, bins=30)
        plt.title('Object Size Distribution')
        plt.xlabel('Normalized Area')
        plt.ylabel('Count')
        
        # Box aspects
        plt.subplot(1, 3, 3)
        plt.hist(self.stats.box_aspects, bins=30)
        plt.title('Object Aspect Ratio Distribution')
        plt.xlabel('Width/Height Ratio')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'box_statistics.png')
        plt.close()
        
    def _plot_quality_metrics(self):
        """Plot image quality metrics"""
        plt.figure(figsize=(15, 5))
        
        # Blur scores
        plt.subplot(1, 3, 1)
        plt.hist(self.stats.blur_scores, bins=30)
        plt.title('Blur Score Distribution')
        plt.xlabel('Blur Score')
        plt.ylabel('Count')
        
        # Brightness
        plt.subplot(1, 3, 2)
        plt.hist(self.stats.brightness_scores, bins=30)
        plt.title('Brightness Distribution')
        plt.xlabel('Mean Brightness')
        plt.ylabel('Count')
        
        # Contrast
        plt.subplot(1, 3, 3)
        plt.hist(self.stats.contrast_scores, bins=30)
        plt.title('Contrast Distribution')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics.png')
        plt.close()
        
    def get_statistics_summary(self) -> Dict:
        """Generate summary of dataset statistics"""
        summary = {
            'dataset_size': {
                'total_images': sum(s['images'] for s in self.stats.split_distribution.values()),
                'total_labels': sum(s['labels'] for s in self.stats.split_distribution.values()),
                'split_distribution': self.stats.split_distribution
            },
            'class_stats': {
                'total_per_class': dict(self.stats.class_counts),
                'distribution_per_split': {
                    split: dict(dist) for split, dist in self.stats.class_per_split.items()
                }
            },
            'image_stats': {
                'size_range': {
                    'min': tuple(np.min(self.stats.image_sizes, axis=0)),
                    'max': tuple(np.max(self.stats.image_sizes, axis=0)),
                    'mean': tuple(np.mean(self.stats.image_sizes, axis=0))
                },
                'aspect_ratio': {
                    'min': min(self.stats.aspect_ratios),
                    'max': max(self.stats.aspect_ratios),
                    'mean': np.mean(self.stats.aspect_ratios)
                },
                'file_size_kb': {
                    'min': min(self.stats.file_sizes),
                    'max': max(self.stats.file_sizes),
                    'mean': np.mean(self.stats.file_sizes)
                }
            },
            'object_stats': {
                'objects_per_image': {
                    'min': min(self.stats.boxes_per_image),
                    'max': max(self.stats.boxes_per_image),
                    'mean': np.mean(self.stats.boxes_per_image)
                },
                'object_size': {
                    'min': min(self.stats.box_sizes),
                    'max': max(self.stats.box_sizes),
                    'mean': np.mean(self.stats.box_sizes)
                },
                'aspect_ratio': {
                    'min': min(self.stats.box_aspects),
                    'max': max(self.stats.box_aspects),
                    'mean': np.mean(self.stats.box_aspects)
                }
            },
            'quality_metrics': {
                'blur_scores': {
                    'min': min(self.stats.blur_scores),
                    'max': max(self.stats.blur_scores),
                    'mean': np.mean(self.stats.blur_scores)
                },
                'brightness': {
                    'min': min(self.stats.brightness_scores),
                    'max': max(self.stats.brightness_scores),
                    'mean': np.mean(self.stats.brightness_scores)
                },
                'contrast': {
                    'min': min(self.stats.contrast_scores),
                    'max': max(self.stats.contrast_scores),
                    'mean': np.mean(self.stats.contrast_scores)
                }
            }
        }
        return summary

    def analyze_class_balance(self) -> Dict:
        """Analyze class balance and imbalance issues"""
        analysis = {
            'class_ratios': {},
            'imbalance_issues': [],
            'recommendations': []
        }
        
        try:
            # Calculate class ratios
            total_objects = sum(self.stats.class_counts.values())
            ideal_ratio = 1.0 / len(self.stats.class_counts)
            
            for cls, count in self.stats.class_counts.items():
                ratio = count / total_objects
                analysis['class_ratios'][cls] = ratio
                
                # Check for imbalance
                if ratio < ideal_ratio * 0.5:
                    analysis['imbalance_issues'].append({
                        'class': cls,
                        'issue': 'Underrepresented',
                        'current_ratio': ratio,
                        'ideal_ratio': ideal_ratio,
                        'difference': ideal_ratio - ratio
                    })
                elif ratio > ideal_ratio * 1.5:
                    analysis['imbalance_issues'].append({
                        'class': cls,
                        'issue': 'Overrepresented',
                        'current_ratio': ratio,
                        'ideal_ratio': ideal_ratio,
                        'difference': ratio - ideal_ratio
                    })
                    
            # Generate recommendations
            if analysis['imbalance_issues']:
                for issue in analysis['imbalance_issues']:
                    if issue['issue'] == 'Underrepresented':
                        target_count = int(total_objects * ideal_ratio)
                        current_count = self.stats.class_counts[issue['class']]
                        needed = target_count - current_count
                        
                        analysis['recommendations'].append({
                            'class': issue['class'],
                            'action': 'Add more samples or augment',
                            'target_count': target_count,
                            'needed_samples': needed
                        })
                    else:
                        analysis['recommendations'].append({
                            'class': issue['class'],
                            'action': 'Consider reducing augmentation'
                        })
                        
        except Exception as e:
            self.logger.error(f"Error analyzing class balance: {str(e)}")
            
        return analysis

    def analyze_quality_issues(self) -> Dict:
        """Analyze potential quality issues in the dataset"""
        analysis = {
            'blur_issues': [],
            'lighting_issues': [],
            'recommendations': []
        }
        
        try:
            # Analyze blur
            blur_threshold = np.mean(self.stats.blur_scores) * 0.5
            for score in self.stats.blur_scores:
                if score < blur_threshold:
                    analysis['blur_issues'].append({
                        'score': score,
                        'threshold': blur_threshold,
                        'severity': 'High' if score < blur_threshold * 0.5 else 'Medium'
                    })
                    
            # Analyze lighting
            brightness_mean = np.mean(self.stats.brightness_scores)
            brightness_std = np.std(self.stats.brightness_scores)
            
            for brightness in self.stats.brightness_scores:
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
                    
            # Generate recommendations
            if len(analysis['blur_issues']) > 0:
                analysis['recommendations'].append({
                    'issue': 'Blur',
                    'action': 'Consider removing or re-capturing blurry images',
                    'affected_count': len(analysis['blur_issues'])
                })
                
            if len(analysis['lighting_issues']) > 0:
                analysis['recommendations'].append({
                    'issue': 'Lighting',
                    'action': 'Consider adjusting lighting conditions or applying normalization',
                    'affected_count': len(analysis['lighting_issues'])
                })
                
        except Exception as e:
            self.logger.error(f"Error analyzing quality issues: {str(e)}")
            
        return analysis

    def get_augmentation_recommendations(self) -> Dict:
        """Generate recommendations for data augmentation"""
        recommendations = {
            'general': [],
            'per_class': {}
        }
        
        try:
            # Analyze class balance
            balance_analysis = self.analyze_class_balance()
            
            # Analyze quality
            quality_analysis = self.analyze_quality_issues()
            
            # General recommendations
            if balance_analysis['imbalance_issues']:
                recommendations['general'].append({
                    'type': 'Class Balance',
                    'description': 'Dataset memiliki ketidakseimbangan kelas',
                    'actions': [
                        'Terapkan augmentasi selektif untuk kelas minoritas',
                        'Pertimbangkan teknik sampling untuk menyeimbangkan kelas'
                    ]
                })
                
            if quality_analysis['blur_issues'] or quality_analysis['lighting_issues']:
                recommendations['general'].append({
                    'type': 'Quality Enhancement',
                    'description': 'Terdapat masalah kualitas gambar',
                    'actions': [
                        'Terapkan augmentasi pencahayaan untuk variasi kondisi',
                        'Pertimbangkan teknik peningkatan kualitas gambar'
                    ]
                })
                
            # Per-class recommendations
            for cls in self.stats.class_counts.keys():
                class_recs = []
                
                # Check representation
                ratio = balance_analysis['class_ratios'].get(cls, 0)
                if ratio < 0.1:  # Significantly underrepresented
                    class_recs.append({
                        'priority': 'High',
                        'action': f'Tambahkan sampel baru atau augmentasi untuk kelas {cls}'
                    })
                elif ratio < 0.2:  # Moderately underrepresented
                    class_recs.append({
                        'priority': 'Medium',
                        'action': f'Pertimbangkan augmentasi ringan untuk kelas {cls}'
                    })
                    
                recommendations['per_class'][cls] = class_recs
                
        except Exception as e:
            self.logger.error(f"Error generating augmentation recommendations: {str(e)}")
            
        return recommendations