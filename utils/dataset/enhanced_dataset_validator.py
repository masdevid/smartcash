"""
File: smartcash/utils/dataset/enhanced_dataset_validator.py
Author: Alfrida Sabar
Deskripsi: Modul validasi dataset yang telah direfaktor dengan pendekatan komponen-komponen terpisah
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import json
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset.dataset_validator_core import DatasetValidatorCore
from smartcash.utils.dataset.dataset_analyzer import DatasetAnalyzer
from smartcash.utils.dataset.dataset_utils import DatasetUtils
from smartcash.utils.dataset.dataset_fixer import DatasetFixer

class EnhancedDatasetValidator:
    """
    Validator dataset yang ditingkatkan dengan kemampuan:
    - Validasi label multilayer
    - Perbaikan otomatis untuk masalah umum
    - Visualisasi masalah
    - Analisis distribusi kelas/layer
    - Error recovery yang kuat
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4
    ):
        """
        Inisialisasi validator dataset yang ditingkatkan.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset
            logger: Logger kustom
            num_workers: Jumlah worker untuk paralelisasi
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.num_workers = num_workers
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Setup komponen
        self.validator_core = DatasetValidatorCore(config, data_dir, logger)
        self.analyzer = DatasetAnalyzer(config, data_dir, logger)
        self.utils = DatasetUtils(logger)
        self.fixer = DatasetFixer(config, data_dir, logger)
        
        # Setup direktori untuk file tidak valid
        self.invalid_dir = self.data_dir / 'invalid'
        
        # Lock untuk thread safety
        self._lock = threading.RLock()
        
    def validate_dataset(
        self,
        split: str = 'train',
        fix_issues: bool = False,
        move_invalid: bool = False,
        visualize: bool = False,
        sample_size: int = 0
    ) -> Dict:
        """
        Validasi dataset untuk satu split.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            fix_issues: Jika True, perbaiki masalah yang ditemukan
            move_invalid: Jika True, pindahkan file tidak valid
            visualize: Jika True, hasilkan visualisasi masalah
            sample_size: Jika > 0, gunakan subset untuk percepatan
            
        Returns:
            Dict hasil validasi
        """
        start_time = time.time()
        
        split_dir = self.data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Pastikan direktori ada
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_dir}")
            return {
                'status': 'error',
                'message': f"Direktori dataset tidak lengkap: {split_dir}",
                'stats': {}
            }
        
        # Setup direktori untuk visualisasi dan file tidak valid jika diperlukan
        vis_dir = None
        if visualize:
            vis_dir = self.data_dir / 'visualizations' / split
            vis_dir.mkdir(parents=True, exist_ok=True)
            
        if move_invalid:
            (self.invalid_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.invalid_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Temukan semua file gambar
        image_files = self.utils.find_image_files(images_dir)
            
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {
                'status': 'warning',
                'message': f"Tidak ada gambar ditemukan di {images_dir}",
                'stats': {
                    'total_images': 0,
                    'total_labels': 0
                }
            }
            
        # Jika sample_size ditentukan, ambil sampel acak
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        # Persiapkan statistik untuk hasil validasi
        validation_stats = self._init_validation_stats()
        validation_stats['total_images'] = len(image_files)
        
        # Proses validasi
        self.logger.info(f"🔍 Memvalidasi dataset {split}: {len(image_files)} gambar")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Siapkan futures
            futures = []
            for img_path in image_files:
                futures.append(
                    executor.submit(
                        self.validator_core.validate_image_label_pair,
                        img_path=img_path,
                        labels_dir=labels_dir
                    )
                )
            
            # Collect results with progress
            results = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="🔍 Validasi Dataset"
            ):
                results.append(future.result())
        
        # Agregasi hasil dan siapkan statistik
        self._aggregate_validation_results(results, validation_stats, fix_issues, labels_dir, visualize, vis_dir)
        
        # Pindahkan file tidak valid jika diminta
        if move_invalid:
            self._move_invalid_files(split, results)
        
        # Catat durasi validasi
        validation_stats['duration'] = time.time() - start_time
        
        # Log ringkasan validasi
        self._log_validation_summary(split, validation_stats)
        
        return validation_stats
    
    def _init_validation_stats(self) -> Dict:
        """Inisialisasi statistik validasi."""
        return {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupt_images': 0,
            'total_labels': 0,
            'valid_labels': 0,
            'invalid_labels': 0,
            'missing_labels': 0,
            'empty_labels': 0,
            'fixed_labels': 0,
            'fixed_coordinates': 0,
            'layer_stats': {layer: 0 for layer in self.validator_core.active_layers},
            'class_stats': {},
            'issues': []
        }
    
    def _aggregate_validation_results(
        self,
        results: List[Dict],
        validation_stats: Dict,
        fix_issues: bool,
        labels_dir: Path,
        visualize: bool,
        vis_dir: Optional[Path]
    ) -> None:
        """
        Agregasi hasil validasi dan update statistik.
        
        Args:
            results: List hasil validasi
            validation_stats: Dict statistik validasi (dimodifikasi inplace)
            fix_issues: Flag untuk perbaiki masalah
            labels_dir: Direktori label
            visualize: Flag untuk buat visualisasi
            vis_dir: Direktori visualisasi
        """
        for result in results:
            # Validasi gambar
            if result.get('image_valid', False):
                validation_stats['valid_images'] += 1
            else:
                validation_stats['invalid_images'] += 1
                
            if result.get('corrupt', False):
                validation_stats['corrupt_images'] += 1
            
            # Validasi label
            if result.get('label_exists', False):
                validation_stats['total_labels'] += 1
                
                if result.get('label_valid', False):
                    validation_stats['valid_labels'] += 1
                    
                    # Update statistik layer
                    for layer, count in result.get('layer_stats', {}).items():
                        if layer in validation_stats['layer_stats']:
                            validation_stats['layer_stats'][layer] += count
                        
                    # Update statistik kelas
                    for cls, count in result.get('class_stats', {}).items():
                        if cls not in validation_stats['class_stats']:
                            validation_stats['class_stats'][cls] = 0
                        validation_stats['class_stats'][cls] += count
                else:
                    validation_stats['invalid_labels'] += 1
                    
                if result.get('empty_label', False):
                    validation_stats['empty_labels'] += 1
                
                # Perbaiki label jika diminta
                if fix_issues and 'fixed_bbox' in result and result['fixed']:
                    label_path = Path(result['label_path'])
                    with open(label_path, 'w') as f:
                        for line in result['fixed_bbox']:
                            f.write(line)
                    validation_stats['fixed_labels'] += 1
                    
                    # Hitung jumlah koordinat yang diperbaiki
                    if 'fixed_coordinates' in result:
                        validation_stats['fixed_coordinates'] += result['fixed_coordinates']
            else:
                validation_stats['missing_labels'] += 1
            
            # Kumpulkan masalah
            if result.get('issues'):
                for issue in result['issues']:
                    validation_stats['issues'].append(
                        f"{Path(result['image_path']).name}: {issue}"
                    )
            
            # Visualisasi jika diminta
            if visualize and vis_dir and result.get('issues') and not result.get('visualized'):
                img_path = Path(result['image_path'])
                self.validator_core.visualize_issues(img_path, result, vis_dir)
                result['visualized'] = True
    
    def _move_invalid_files(
        self,
        split: str,
        validation_results: List[Dict]
    ) -> Dict[str, int]:
        """
        Pindahkan file tidak valid ke direktori terpisah.
        
        Args:
            split: Split dataset
            validation_results: Hasil validasi
            
        Returns:
            Dict statistik pemindahan
        """
        self.logger.info(f"🔄 Memindahkan file tidak valid ke {self.invalid_dir}...")
        
        # Filter file tidak valid
        invalid_images = []
        invalid_labels = []
        
        for result in validation_results:
            # Skip yang valid
            if result.get('status') == 'valid':
                continue
                
            # Gambar rusak atau tidak valid
            if not result.get('image_valid', False) or result.get('corrupt', False):
                invalid_images.append(Path(result['image_path']))
            
            # Label tidak valid
            if result.get('label_exists', False) and not result.get('label_valid', False):
                invalid_labels.append(Path(result['label_path']))
        
        # Pindahkan file
        img_stats = self.utils.move_invalid_files(
            self.data_dir / split / 'images',
            self.invalid_dir / split / 'images',
            invalid_images
        )
        
        label_stats = self.utils.move_invalid_files(
            self.data_dir / split / 'labels',
            self.invalid_dir / split / 'labels',
            invalid_labels
        )
        
        # Gabungkan statistik
        stats = {
            'moved_images': img_stats['moved'],
            'moved_labels': label_stats['moved'],
            'errors': img_stats['errors'] + label_stats['errors']
        }
        
        self.logger.success(
            f"✅ Pemindahan file tidak valid selesai:\n"
            f"   • Gambar dipindahkan: {stats['moved_images']}\n"
            f"   • Label dipindahkan: {stats['moved_labels']}\n"
            f"   • Error: {stats['errors']}"
        )
        
        return stats
    
    def _log_validation_summary(self, split: str, validation_stats: Dict) -> None:
        """
        Log ringkasan hasil validasi.
        
        Args:
            split: Split dataset
            validation_stats: Statistik validasi
        """
        self.logger.info(
            f"✅ Ringkasan Validasi Dataset {split} ({validation_stats['duration']:.1f} detik):\n"
            f"📸 Total Gambar: {validation_stats['total_images']}\n"
            f"   • Valid: {validation_stats['valid_images']}\n"
            f"   • Tidak Valid: {validation_stats['invalid_images']}\n"
            f"   • Corrupt: {validation_stats['corrupt_images']}\n"
            f"📋 Total Label: {validation_stats['total_labels']}\n"
            f"   • Valid: {validation_stats['valid_labels']}\n"
            f"   • Tidak Valid: {validation_stats['invalid_labels']}\n"
            f"   • Label Hilang: {validation_stats['missing_labels']}\n"
            f"   • Label Kosong: {validation_stats['empty_labels']}\n"
            f"🔧 Perbaikan:\n"
            f"   • Label Diperbaiki: {validation_stats['fixed_labels']}\n"
            f"   • Koordinat Diperbaiki: {validation_stats['fixed_coordinates']}"
        )
        
        # Log statistik per layer
        self.logger.info("📊 Distribusi Layer:")
        for layer, count in validation_stats['layer_stats'].items():
            if count > 0:
                percentage = 0
                if validation_stats['valid_labels'] > 0:
                    percentage = (count / validation_stats['valid_labels']) * 100
                self.logger.info(f"   • {layer}: {count} objek ({percentage:.1f}%)")
        
        # Log statistik kelas (top 10)
        if validation_stats['class_stats']:
            top_classes = sorted(
                validation_stats['class_stats'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            self.logger.info("📊 Top 10 Kelas:")
            for cls, count in top_classes:
                percentage = 0
                if validation_stats['valid_labels'] > 0:
                    percentage = (count / validation_stats['valid_labels']) * 100
                self.logger.info(f"   • {cls}: {count} ({percentage:.1f}%)")
    
    def analyze_dataset(
        self,
        split: str = 'train',
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis mendalam tentang dataset.
        
        Args:
            split: Split dataset
            sample_size: Jika > 0, gunakan sampel
            
        Returns:
            Dict hasil analisis
        """
        # Jalankan validasi tanpa perbaikan atau pemindahan
        validation_results = self.validate_dataset(
            split=split,
            fix_issues=False,
            move_invalid=False,
            visualize=False,
            sample_size=sample_size
        )
        
        # Tambahkan analisis lebih lanjut
        analysis = {
            'validation': validation_results,
            'image_size_distribution': self.analyzer.analyze_image_sizes(split, sample_size),
            'class_balance': self.analyzer.analyze_class_balance(validation_results),
            'layer_balance': self.analyzer.analyze_layer_balance(validation_results),
            'bbox_statistics': self.analyzer.analyze_bbox_statistics(split, sample_size)
        }
        
        # Log hasil analisis
        self.logger.info(
            f"📊 Analisis Dataset {split}:\n"
            f"   • Ketidakseimbangan kelas: {analysis['class_balance']['imbalance_score']:.2f}/10\n"
            f"   • Ketidakseimbangan layer: {analysis['layer_balance']['imbalance_score']:.2f}/10\n"
            f"   • Ukuran gambar yang dominan: {analysis['image_size_distribution']['dominant_size']}\n"
            f"   • Rasio aspek dominan: {analysis['image_size_distribution']['dominant_aspect_ratio']}\n"
            f"   • Ukuran bbox rata-rata: {analysis['bbox_statistics']['mean_width']:.2f}x{analysis['bbox_statistics']['mean_height']:.2f}"
        )
        
        return analysis
    
    def fix_dataset(
        self,
        split: str = 'train',
        fix_coordinates: bool = True,
        fix_labels: bool = True,
        fix_images: bool = False,
        backup: bool = True
    ) -> Dict:
        """
        Perbaiki masalah umum dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            fix_coordinates: Perbaiki koordinat yang tidak valid
            fix_labels: Perbaiki format label yang tidak valid
            fix_images: Coba perbaiki gambar yang rusak
            backup: Buat backup sebelum perbaikan
            
        Returns:
            Dict statistik perbaikan
        """
        return self.fixer.fix_dataset(
            split=split,
            fix_coordinates=fix_coordinates,
            fix_labels=fix_labels,
            fix_images=fix_images,
            backup=backup
        )