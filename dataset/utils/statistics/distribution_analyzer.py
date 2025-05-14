"""
File: smartcash/dataset/utils/statistics/distribution_analyzer.py
Deskripsi: Utilitas untuk menganalisis distribusi statistik dataset secara menyeluruh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.dataset.utils.statistics.class_stats import ClassStatistics
from smartcash.dataset.utils.statistics.image_stats import ImageStatistics



class DistributionAnalyzer:
    """Utilitas untuk analisis komprehensif distribusi dataset."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None, num_workers: int = 4):
        """
        Inisialisasi DistributionAnalyzer.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("distribution_analyzer")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        self.class_stats = ClassStatistics(config, data_dir, logger)
        self.image_stats = ImageStatistics(config, data_dir, logger, num_workers)
        
        self.logger.info(f"📊 DistributionAnalyzer diinisialisasi dengan data_dir: {self.data_dir}")
    
    def analyze_dataset(self, splits: List[str] = None, sample_size: int = 0) -> Dict[str, Any]:
        """
        Lakukan analisis komprehensif pada dataset.
        
        Args:
            splits: Daftar split dataset yang akan dianalisis (default: train, valid, test)
            sample_size: Jumlah sampel per split (0 = semua)
            
        Returns:
            Dictionary dengan hasil analisis menyeluruh
        """
        if splits is None:
            splits = DEFAULT_SPLITS
            
        self.logger.info(f"🔍 Memulai analisis komprehensif untuk {len(splits)} split dataset")
        
        results = {
            'split_stats': {},
            'class_distribution': {},
            'size_distribution': {},
            'quality_stats': {},
            'cross_split_stats': {
                'class_consistency': {},
                'size_consistency': {}
            },
            'suggestions': []
        }
        
        # Analisis per split
        for split in splits:
            split_path = self.utils.get_split_path(split)
            if not split_path.exists():
                self.logger.warning(f"⚠️ Split '{split}' tidak ditemukan, dilewati")
                continue
                
            self.logger.info(f"📊 Menganalisis split '{split}'...")
            
            # Analisis kelas
            class_result = self.class_stats.analyze_distribution(split, sample_size)
            
            # Analisis ukuran gambar
            size_result = self.image_stats.analyze_image_sizes(split, sample_size)
            
            # Analisis kualitas gambar (lebih ringan, gunakan sampel yang lebih kecil)
            quality_sample = min(sample_size if sample_size > 0 else 100, 100)
            quality_result = self.image_stats.analyze_image_quality(split, quality_sample)
            
            # Simpan hasil per split
            results['split_stats'][split] = {
                'class_count': class_result.get('class_count', 0),
                'total_objects': class_result.get('total_objects', 0),
                'imbalance_score': class_result.get('imbalance_score', 0),
                'dominant_size': size_result.get('dominant_size', 'unknown'),
                'quality_distribution': quality_result.get('quality_categories', {})
            }
            
            results['class_distribution'][split] = class_result.get('counts', {})
            results['size_distribution'][split] = size_result.get('size_categories', {})
            results['quality_stats'][split] = {
                'blur': quality_result.get('blur_stats', {}),
                'contrast': quality_result.get('contrast_stats', {}),
                'noise': quality_result.get('noise_stats', {})
            }
        
        # Analisis konsistensi antar split
        if len(splits) > 1:
            self._analyze_cross_split_consistency(results)
            
        # Buat rekomendasi
        results['suggestions'] = self._generate_suggestions(results)
        
        # Log ringkasan
        self._log_summary(results)
        
        return results
    
    def _analyze_cross_split_consistency(self, results: Dict[str, Any]) -> None:
        """
        Analisis konsistensi antar split dataset.
        
        Args:
            results: Dictionary hasil analisis yang akan diupdate
        """
        # Analisis konsistensi distribusi kelas antar split
        if len(results['class_distribution']) > 1:
            # Hitung overlap kelas antar split
            all_classes = set()
            for split_classes in results['class_distribution'].values():
                all_classes.update(split_classes.keys())
                
            for split, classes in results['class_distribution'].items():
                coverage = len(classes) / len(all_classes) if all_classes else 0
                results['cross_split_stats']['class_consistency'][split] = {
                    'coverage': coverage,
                    'unique_classes': len(set(classes.keys()) - set().union(*[
                        set(s.keys()) for s, d in results['class_distribution'].items() if s != split
                    ]))
                }
        
        # Analisis konsistensi ukuran gambar antar split
        if len(results['size_distribution']) > 1:
            all_sizes = results['size_distribution'].keys()
            for split, size_data in results['size_distribution'].items():
                other_splits = [s for s in all_sizes if s != split]
                
                # Hitung variasi ukuran dominan
                dominant_sizes = []
                for s in all_sizes:
                    if s in results['split_stats'] and 'dominant_size' in results['split_stats'][s]:
                        dominant_sizes.append(results['split_stats'][s]['dominant_size'])
                
                results['cross_split_stats']['size_consistency'][split] = {
                    'size_variation': len(set(dominant_sizes))
                }
    
    def _generate_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """
        Buat rekomendasi berdasarkan hasil analisis.
        
        Args:
            results: Dictionary hasil analisis
            
        Returns:
            Daftar rekomendasi
        """
        suggestions = []
        
        # Periksa ketidakseimbangan kelas
        for split, stats in results['split_stats'].items():
            if stats.get('imbalance_score', 0) > 5.0:
                suggestions.append(
                    f"🔄 Split '{split}' memiliki ketidakseimbangan kelas yang tinggi " 
                    f"(skor: {stats['imbalance_score']:.1f}/10). Pertimbangkan untuk melakukan "
                    f"oversampling atau undersampling."
                )
        
        # Periksa konsistensi antar split
        if 'cross_split_stats' in results and 'class_consistency' in results['cross_split_stats']:
            for split, consistency in results['cross_split_stats']['class_consistency'].items():
                if consistency.get('coverage', 1.0) < 0.8:
                    suggestions.append(
                        f"⚠️ Split '{split}' hanya mencakup {consistency['coverage']*100:.1f}% dari "
                        f"semua kelas. Pastikan semua split memiliki representasi kelas yang seimbang."
                    )
                
                if consistency.get('unique_classes', 0) > 0:
                    suggestions.append(
                        f"⚠️ Split '{split}' memiliki {consistency['unique_classes']} kelas unik yang "
                        f"tidak ada di split lain. Ini bisa menyebabkan masalah generalisasi model."
                    )
        
        # Periksa konsistensi ukuran gambar
        if 'cross_split_stats' in results and 'size_consistency' in results['cross_split_stats']:
            for split, consistency in results['cross_split_stats']['size_consistency'].items():
                if consistency.get('size_variation', 1) > 1:
                    suggestions.append(
                        f"🖼️ Terdapat variasi ukuran gambar dominan antar split. "
                        f"Pertimbangkan untuk melakukan resize semua gambar ke ukuran yang sama."
                    )
                    break  # Cukup satu rekomendasi untuk ukuran
        
        # Periksa kualitas gambar
        for split, quality in results['quality_stats'].items():
            blur_stats = quality.get('blur', {})
            if blur_stats and 'median' in blur_stats and blur_stats['median'] < 50:
                suggestions.append(
                    f"🔍 Split '{split}' memiliki banyak gambar blur (median skor: {blur_stats['median']:.1f}). "
                    f"Pertimbangkan untuk melakukan pemfilteran atau perbaikan gambar."
                )
                
            contrast_stats = quality.get('contrast', {})
            if contrast_stats and 'median' in contrast_stats and contrast_stats['median'] < 40:
                suggestions.append(
                    f"🎨 Split '{split}' memiliki banyak gambar dengan kontras rendah "
                    f"(median skor: {contrast_stats['median']:.1f}). "
                    f"Pertimbangkan untuk meningkatkan kontras gambar."
                )
        
        return suggestions
    
    def _log_summary(self, results: Dict[str, Any]) -> None:
        """
        Log ringkasan hasil analisis.
        
        Args:
            results: Dictionary hasil analisis
        """
        self.logger.info(f"✅ Analisis dataset selesai")
        
        # Log statistik per split
        for split, stats in results['split_stats'].items():
            self.logger.info(
                f"📊 Split '{split}':\n"
                f"   • Jumlah kelas: {stats.get('class_count', 0)}\n"
                f"   • Total objek: {stats.get('total_objects', 0)}\n"
                f"   • Skor ketidakseimbangan: {stats.get('imbalance_score', 0):.1f}/10\n"
                f"   • Ukuran dominan: {stats.get('dominant_size', 'unknown')}"
            )
        
        # Log rekomendasi
        if results['suggestions']:
            self.logger.info("💡 Rekomendasi:")
            for suggestion in results['suggestions']:
                self.logger.info(f"   • {suggestion}")
        else:
            self.logger.info("💡 Tidak ada rekomendasi khusus - dataset terlihat seimbang")