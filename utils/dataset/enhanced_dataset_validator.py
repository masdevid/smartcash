"""
File: smartcash/utils/dataset/enhanced_dataset_validator.py
Author: Alfrida Sabar
Deskripsi: Modul validasi dataset yang dioptimasi dengan menghilangkan duplikasi kode
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset.dataset_validator_core import DatasetValidatorCore
from smartcash.utils.dataset.dataset_analyzer import DatasetAnalyzer
from smartcash.utils.dataset.dataset_utils import DatasetUtils
from smartcash.utils.dataset.dataset_fixer import DatasetFixer

class EnhancedDatasetValidator:
    """
    Validator dataset yang ditingkatkan untuk validasi label multilayer, perbaikan otomatis,
    visualisasi masalah, analisis distribusi, dan error recovery yang kuat.
    """
    
    def __init__(self, config: Dict, data_dir: Optional[Union[str, Path]] = None,
                logger: Optional[SmartCashLogger] = None, num_workers: int = 4):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.num_workers = num_workers
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        self.utils = DatasetUtils(config=config, data_dir=str(self.data_dir), logger=logger)
        self.validator_core = DatasetValidatorCore(config, data_dir, logger)
        self.analyzer = DatasetAnalyzer(config, data_dir, logger)
        self.fixer = DatasetFixer(config, data_dir, logger)
        self.invalid_dir = self.data_dir / 'invalid'
        self._lock = threading.RLock()
        
    def validate_dataset(self, split: str = 'train', fix_issues: bool = False,
                       move_invalid: bool = False, visualize: bool = False, sample_size: int = 0) -> Dict:
        """Validasi dataset untuk satu split."""
        start_time = time.time()
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori dataset tidak lengkap: {split_path}", 'stats': {}}
        
        # Setup direktori
        vis_dir = (self.data_dir / 'visualizations' / split) if visualize else None
        if vis_dir: vis_dir.mkdir(parents=True, exist_ok=True)
        if move_invalid:
            (self.invalid_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.invalid_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Cari file gambar
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'warning', 'message': f"Tidak ada gambar ditemukan di {images_dir}", 
                   'stats': {'total_images': 0, 'total_labels': 0}}
        
        # Gunakan sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        # Statistik validasi
        validation_stats = self._init_validation_stats()
        validation_stats['total_images'] = len(image_files)
        
        self.logger.info(f"🔍 Memvalidasi dataset {split}: {len(image_files)} gambar")
        
        # Validasi paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.validator_core.validate_image_label_pair, img_path, labels_dir) 
                      for img_path in image_files]
            
            # Collect results with progress
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="🔍 Validasi Dataset"):
                results.append(future.result())
        
        # Agregasi hasil
        self._aggregate_validation_results(results, validation_stats, fix_issues, labels_dir, visualize, vis_dir)
        
        # Pindahkan file tidak valid jika diminta
        if move_invalid: self._move_invalid_files(split, results)
        
        # Catat durasi dan log hasil
        validation_stats['duration'] = time.time() - start_time
        self._log_validation_summary(split, validation_stats)
        
        return validation_stats
    
    def _init_validation_stats(self) -> Dict:
        """Inisialisasi statistik validasi."""
        return {
            'total_images': 0, 'valid_images': 0, 'invalid_images': 0, 'corrupt_images': 0,
            'total_labels': 0, 'valid_labels': 0, 'invalid_labels': 0, 'missing_labels': 0, 'empty_labels': 0,
            'fixed_labels': 0, 'fixed_coordinates': 0,
            'layer_stats': {layer: 0 for layer in self.validator_core.active_layers},
            'class_stats': {}, 'issues': []
        }
    
    def _aggregate_validation_results(self, results: List[Dict], validation_stats: Dict, fix_issues: bool,
                                    labels_dir: Path, visualize: bool, vis_dir: Optional[Path]) -> None:
        """Agregasi hasil validasi dan update statistik."""
        for result in results:
            # Validasi gambar
            if result.get('image_valid', False): validation_stats['valid_images'] += 1
            else: validation_stats['invalid_images'] += 1
            if result.get('corrupt', False): validation_stats['corrupt_images'] += 1
            
            # Validasi label
            if result.get('label_exists', False):
                validation_stats['total_labels'] += 1
                
                if result.get('label_valid', False):
                    validation_stats['valid_labels'] += 1
                    
                    # Update statistik layer & kelas
                    for layer, count in result.get('layer_stats', {}).items():
                        if layer in validation_stats['layer_stats']: 
                            validation_stats['layer_stats'][layer] += count
                        
                    for cls, count in result.get('class_stats', {}).items():
                        validation_stats['class_stats'][cls] = validation_stats['class_stats'].get(cls, 0) + count
                else:
                    validation_stats['invalid_labels'] += 1
                    
                if result.get('empty_label', False): validation_stats['empty_labels'] += 1
                
                # Perbaiki label jika diminta
                if fix_issues and 'fixed_bbox' in result and result['fixed']:
                    label_path = Path(result['label_path'])
                    with open(label_path, 'w') as f:
                        for line in result['fixed_bbox']: f.write(line)
                    validation_stats['fixed_labels'] += 1
                    
                    # Hitung koordinat yang diperbaiki
                    if 'fixed_coordinates' in result:
                        validation_stats['fixed_coordinates'] += result['fixed_coordinates']
            else:
                validation_stats['missing_labels'] += 1
            
            # Kumpulkan masalah
            if result.get('issues'):
                for issue in result['issues']:
                    validation_stats['issues'].append(f"{Path(result['image_path']).name}: {issue}")
            
            # Visualisasi jika diminta
            if visualize and vis_dir and result.get('issues') and not result.get('visualized'):
                self.validator_core.visualize_issues(Path(result['image_path']), result, vis_dir)
                result['visualized'] = True
    
    def _move_invalid_files(self, split: str, validation_results: List[Dict]) -> Dict[str, int]:
        """Pindahkan file tidak valid ke direktori terpisah."""
        self.logger.info(f"🔄 Memindahkan file tidak valid ke {self.invalid_dir}...")
        
        # Filter file tidak valid
        invalid_images = [Path(r['image_path']) for r in validation_results 
                        if r.get('status') != 'valid' and 
                           (not r.get('image_valid', False) or r.get('corrupt', False))]
                           
        invalid_labels = [Path(r['label_path']) for r in validation_results 
                        if r.get('status') != 'valid' and 
                           r.get('label_exists', False) and not r.get('label_valid', False)]
        
        # Pindahkan file
        split_path = self.utils.get_split_path(split)
        img_stats = self.utils.move_invalid_files(
            split_path / 'images', self.invalid_dir / split / 'images', invalid_images)
        
        label_stats = self.utils.move_invalid_files(
            split_path / 'labels', self.invalid_dir / split / 'labels', invalid_labels)
        
        # Statistik gabungan
        stats = {
            'moved_images': img_stats['moved'], 'moved_labels': label_stats['moved'],
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
        """Log ringkasan hasil validasi."""
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
                pct = (count / max(1, validation_stats['valid_labels'])) * 100
                self.logger.info(f"   • {layer}: {count} objek ({pct:.1f}%)")
        
        # Log statistik kelas (top 10)
        if validation_stats['class_stats']:
            top_classes = sorted(validation_stats['class_stats'].items(), key=lambda x: x[1], reverse=True)[:10]
            
            self.logger.info("📊 Top 10 Kelas:")
            for cls, count in top_classes:
                pct = (count / max(1, validation_stats['valid_labels'])) * 100
                self.logger.info(f"   • {cls}: {count} ({pct:.1f}%)")
    
    def analyze_dataset(self, split: str = 'train', sample_size: int = 0, detailed: bool = True) -> Dict:
        """Analisis mendalam tentang dataset."""
        # Validasi tanpa perbaikan atau pemindahan
        validation_results = self.validate_dataset(
            split=split, fix_issues=False, move_invalid=False, visualize=False, sample_size=sample_size)
        
        # Analisis lanjutan
        analysis = {
            'validation': validation_results,
            'image_size_distribution': self.analyzer.analyze_image_sizes(split, sample_size),
            'class_balance': self.analyzer.analyze_class_balance(validation_results),
            'layer_balance': self.analyzer.analyze_layer_balance(validation_results)
        }
        
        # Analisis bbox jika detailed
        if detailed: analysis['bbox_statistics'] = self.analyzer.analyze_bbox_statistics(split, sample_size)
        
        # Log hasil
        self.logger.info(
            f"📊 Analisis Dataset {split}:\n"
            f"   • Ketidakseimbangan kelas: {analysis['class_balance']['imbalance_score']:.2f}/10\n"
            f"   • Ketidakseimbangan layer: {analysis['layer_balance']['imbalance_score']:.2f}/10\n"
            f"   • Ukuran gambar dominan: {analysis['image_size_distribution']['dominant_size']}"
        )
        
        if detailed and 'bbox_statistics' in analysis:
            self.logger.info(f"   • Ukuran bbox rata-rata: {analysis['bbox_statistics'].get('mean_width', 0):.2f}x{analysis['bbox_statistics'].get('mean_height', 0):.2f}")
        
        return analysis
    
    def fix_dataset(self, split: str = 'train', fix_coordinates: bool = True,
                   fix_labels: bool = True, fix_images: bool = False, backup: bool = True) -> Dict:
        """Perbaiki masalah umum dalam dataset."""
        return self.fixer.fix_dataset(
            split=split, fix_coordinates=fix_coordinates, fix_labels=fix_labels, 
            fix_images=fix_images, backup=backup)
        
    def get_valid_files(self, data_dir: str, split: str, check_images: bool = True, check_labels: bool = True) -> List[Dict]:
        """Dapatkan daftar file valid dalam dataset."""
        split_path = Path(data_dir) / split if split else Path(data_dir)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not images_dir.exists():
            self.logger.warning(f"⚠️ Direktori gambar tidak ditemukan: {images_dir}")
            return []
            
        # Cari file gambar
        image_files = self.utils.find_image_files(images_dir, with_labels=False)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return []
            
        # Validasi file
        valid_files = []
        for img_path in tqdm(image_files, desc=f"Mencari file valid di {split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Cek kriteria validitas
            is_valid = True
            if check_labels and not label_path.exists():
                is_valid = False
            elif check_labels:
                available_layers = self.utils.get_available_layers(label_path)
                if not available_layers: is_valid = False
            
            if check_images and is_valid:
                try:
                    img = self.utils.load_image(img_path)
                    if img is None: is_valid = False
                except Exception: is_valid = False
            
            # Tambahkan ke hasil jika valid
            if is_valid:
                valid_files.append({
                    'image_path': str(img_path),
                    'label_path': str(label_path),
                    'available_layers': self.utils.get_available_layers(label_path) if check_labels else []
                })
        
        self.logger.info(f"✅ Ditemukan {len(valid_files)} file valid dari {len(image_files)} total")
        return valid_files