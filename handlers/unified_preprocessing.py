# File: smartcash/handlers/unified_preprocessing_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler terintegrasi untuk seluruh pipeline preprocessing data

import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.enhanced_cache import EnhancedCache
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.optimized_augmentation import OptimizedAugmentation
from smartcash.utils.enhanced_dataset_validator import EnhancedDatasetValidator

class UnifiedPreprocessingHandler:
    """
    Handler terintegrasi untuk seluruh pipeline preprocessing data:
    - Validasi dataset
    - Perbaikan dataset
    - Augmentasi dataset
    - Monitoring kualitas data
    
    Menyederhanakan alur kerja preprocessing dengan mengelola semua komponen
    dalam satu handler yang kohesif.
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4,
        cache_size_gb: float = 1.0
    ):
        """
        Inisialisasi handler preprocessing terintegrasi.
        
        Args:
            config_path: Path ke file konfigurasi
            output_dir: Direktori output (jika None, gunakan dari config)
            logger: Logger kustom
            num_workers: Jumlah worker untuk paralelisasi
            cache_size_gb: Ukuran cache dalam GB
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.num_workers = num_workers
        
        # Load konfigurasi
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Setup direktori
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Siapkan komponen utama
        self.logger.info("🏗️ Menyiapkan komponen preprocessing...")
        
        # Layer config manager 
        self.layer_config = get_layer_config()
        self.active_layers = self.config.get('layers', ['banknote'])
        
        # Cache sistem
        self.cache = EnhancedCache(
            cache_dir=".cache/smartcash",
            max_size_gb=cache_size_gb,
            logger=self.logger
        )
        
        # Dataset validator
        self.validator = EnhancedDatasetValidator(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger,
            num_workers=num_workers
        )
        
        # Augmentasi
        self.augmentor = OptimizedAugmentation(
            config=self.config,
            output_dir=self.output_dir,
            logger=self.logger,
            num_workers=num_workers
        )
        
        self.logger.success(f"✅ Handler preprocessing siap dengan {len(self.active_layers)} layer aktif")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load konfigurasi dari file YAML.
        
        Args:
            config_path: Path ke file konfigurasi
            
        Returns:
            Dict konfigurasi
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Log konfigurasi penting
            self.logger.info(
                f"📥 Konfigurasi dimuat dari {config_path}"
            )
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi: {str(e)}")
            # Fallback ke konfigurasi minimal
            return {
                'data_dir': 'data',
                'layers': ['banknote']
            }
    
    def run_full_pipeline(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        validate_only: bool = False,
        fix_issues: bool = True,
        augment_data: bool = True,
        aug_types: List[str] = ['combined'],
        backup: bool = True
    ) -> Dict:
        """
        Jalankan seluruh pipeline preprocessing.
        
        Args:
            splits: List split dataset yang akan diproses
            validate_only: Jika True, hanya validasi tanpa perbaikan
            fix_issues: Jika True, perbaiki masalah yang ditemukan
            augment_data: Jika True, lakukan augmentasi data
            aug_types: Tipe augmentasi yang digunakan
            backup: Buat backup sebelum perbaikan
            
        Returns:
            Dict statistik pipeline
        """
        pipeline_start = time.time()
        
        pipeline_stats = {
            'validation': {},
            'fixes': {},
            'augmentation': {},
            'duration': 0.0
        }
        
        # 1. Validasi semua split
        self.logger.start("🔍 Langkah 1: Validasi Dataset")
        
        for split in splits:
            if not (self.data_dir / split).exists():
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan di {self.data_dir}, melewati...")
                continue
                
            self.logger.info(f"🔍 Memvalidasi split: {split}")
            
            try:
                validation_stats = self.validator.validate_dataset(
                    split=split,
                    fix_issues=False,  # Belum perbaiki di sini
                    move_invalid=False
                )
                
                pipeline_stats['validation'][split] = validation_stats
                
                # Check kualitas data
                valid_labels = validation_stats.get('valid_labels', 0)
                invalid_labels = validation_stats.get('invalid_labels', 0) + validation_stats.get('missing_labels', 0)
                total_labels = valid_labels + invalid_labels
                
                if total_labels > 0:
                    quality_score = (valid_labels / total_labels) * 100
                    pipeline_stats['validation'][f'{split}_quality'] = quality_score
                    
                    if quality_score < 70:
                        self.logger.warning(
                            f"⚠️ Kualitas data {split} kurang baik: {quality_score:.1f}%"
                        )
                    else:
                        self.logger.info(
                            f"✅ Kualitas data {split}: {quality_score:.1f}%"
                        )
            except Exception as e:
                self.logger.error(f"❌ Error validasi {split}: {str(e)}")
                pipeline_stats['validation'][split] = {'error': str(e)}
        
        # 2. Perbaiki masalah jika diminta
        if fix_issues and not validate_only:
            self.logger.start("🔧 Langkah 2: Perbaikan Dataset")
            
            for split in splits:
                if not (self.data_dir / split).exists():
                    continue
                    
                # Periksa apakah perlu perbaikan
                validation = pipeline_stats['validation'].get(split, {})
                invalid_count = (
                    validation.get('invalid_labels', 0) + 
                    validation.get('missing_labels', 0) +
                    validation.get('fixed_coordinates', 0)
                )
                
                if invalid_count > 0:
                    self.logger.info(f"🔧 Memperbaiki split: {split} ({invalid_count} masalah)")
                    
                    try:
                        fix_stats = self.validator.fix_dataset(
                            split=split,
                            fix_coordinates=True,
                            fix_labels=True,
                            backup=backup
                        )
                        
                        pipeline_stats['fixes'][split] = fix_stats
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error perbaikan {split}: {str(e)}")
                        pipeline_stats['fixes'][split] = {'error': str(e)}
                else:
                    self.logger.info(f"✅ Split {split} tidak memerlukan perbaikan")
                    pipeline_stats['fixes'][split] = {'status': 'not_needed'}
            
            # Re-validasi dataset yang telah diperbaiki
            self.logger.info("🔍 Memvalidasi ulang dataset yang telah diperbaiki...")
            
            for split in splits:
                if not (self.data_dir / split).exists() or split not in pipeline_stats['fixes']:
                    continue
                    
                try:
                    revalidation = self.validator.validate_dataset(
                        split=split,
                        fix_issues=False,
                        move_invalid=False
                    )
                    
                    pipeline_stats['validation'][f"{split}_after_fix"] = revalidation
                    
                    # Bandingkan dengan validasi sebelumnya
                    if split in pipeline_stats['validation']:
                        before = pipeline_stats['validation'][split]
                        after = revalidation
                        
                        invalid_before = before.get('invalid_labels', 0) + before.get('missing_labels', 0)
                        invalid_after = after.get('invalid_labels', 0) + after.get('missing_labels', 0)
                        
                        if invalid_after < invalid_before:
                            improvement = invalid_before - invalid_after
                            self.logger.success(
                                f"✅ Perbaikan {split} berhasil: {improvement} masalah teratasi"
                            )
                        else:
                            self.logger.warning(
                                f"⚠️ Perbaikan {split} tidak efektif"
                            )
                except Exception as e:
                    self.logger.error(f"❌ Error validasi ulang {split}: {str(e)}")
        
        # 3. Augmentasi data jika diminta
        if augment_data and not validate_only:
            self.logger.start("🎨 Langkah 3: Augmentasi Dataset")
            
            # Hanya augmentasi split train
            train_split = 'train'
            if train_split in splits and (self.data_dir / train_split).exists():
                self.logger.info(
                    f"🎨 Augmentasi split {train_split} dengan metode: {', '.join(aug_types)}"
                )
                
                try:
                    aug_stats = self.augmentor.augment_dataset(
                        split=train_split,
                        augmentation_types=aug_types,
                        num_variations=2,  # 2 variasi per tipe
                        output_prefix='aug',
                        validate_results=True
                    )
                    
                    pipeline_stats['augmentation'][train_split] = aug_stats
                    
                    # Log statistik augmentasi
                    if aug_stats.get('augmented', 0) > 0:
                        self.logger.success(
                            f"✅ Augmentasi berhasil: {aug_stats.get('augmented', 0)} gambar baru"
                        )
                except Exception as e:
                    self.logger.error(f"❌ Error augmentasi {train_split}: {str(e)}")
                    pipeline_stats['augmentation'][train_split] = {'error': str(e)}
            else:
                self.logger.warning(f"⚠️ Split {train_split} tidak ditemukan, melewati augmentasi")
                
            # Validasi hasil augmentasi
            if train_split in pipeline_stats['augmentation']:
                self.logger.info(f"🔍 Memvalidasi hasil augmentasi...")
                
                try:
                    post_aug_validation = self.validator.validate_dataset(
                        split=train_split,
                        fix_issues=False,
                        move_invalid=False
                    )
                    
                    pipeline_stats['validation'][f"{train_split}_after_aug"] = post_aug_validation
                    
                    # Bandingkan dengan validasi sebelumnya
                    if train_split in pipeline_stats['validation']:
                        before = pipeline_stats['validation'][train_split]
                        after = post_aug_validation
                        
                        before_images = before.get('total_images', 0)
                        after_images = after.get('total_images', 0)
                        
                        aug_count = after_images - before_images
                        aug_percent = (aug_count / max(1, before_images)) * 100
                        
                        self.logger.info(
                            f"📊 Hasil augmentasi: {aug_count} gambar baru (+{aug_percent:.1f}%)"
                        )
                except Exception as e:
                    self.logger.error(f"❌ Error validasi hasil augmentasi: {str(e)}")
        
        # 4. Finalisasi dan pelaporan
        pipeline_duration = time.time() - pipeline_start
        pipeline_stats['duration'] = pipeline_duration
        
        self.logger.success(
            f"✨ Pipeline preprocessing selesai dalam {pipeline_duration:.1f} detik"
        )
        
        return pipeline_stats
    
    def analyze_dataset_quality(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis mendalam tentang kualitas dataset.
        
        Args:
            splits: List split dataset
            sample_size: Jika > 0, gunakan sampel untuk percepatan
            
        Returns:
            Dict hasil analisis
        """
        analysis_start = time.time()
        
        analysis_results = {
            'quality_scores': {},
            'class_distribution': {},
            'layer_distribution': {},
            'bbox_statistics': {},
            'image_size_stats': {},
            'duration': 0.0,
            'recommendations': []
        }
        
        self.logger.start("🔍 Analisis Kualitas Dataset")
        
        for split in splits:
            if not (self.data_dir / split).exists():
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan, melewati...")
                continue
                
            self.logger.info(f"🔍 Menganalisis split: {split}")
            
            try:
                # Jalankan analisis validator
                split_analysis = self.validator.analyze_dataset(
                    split=split,
                    sample_size=sample_size
                )
                
                # Ekstrak metrik utama
                class_balance = split_analysis.get('class_balance', {})
                layer_balance = split_analysis.get('layer_balance', {})
                image_sizes = split_analysis.get('image_size_distribution', {})
                bbox_stats = split_analysis.get('bbox_statistics', {})
                
                # Hitung skor kualitas (0-100)
                quality_score = 100
                
                # Penalti untuk ketidakseimbangan kelas
                class_imbalance = class_balance.get('imbalance_score', 10)
                quality_score -= class_imbalance * 2  # Penalti maksimum 20 poin
                
                # Penalti untuk ketidakseimbangan layer
                layer_imbalance = layer_balance.get('imbalance_score', 10)
                quality_score -= layer_imbalance * 2  # Penalti maksimum 20 poin
                
                # Penalti untuk validasi dasar
                validation = split_analysis.get('validation', {})
                total_images = validation.get('total_images', 0)
                invalid_images = validation.get('invalid_images', 0)
                missing_labels = validation.get('missing_labels', 0)
                
                if total_images > 0:
                    invalid_pct = ((invalid_images + missing_labels) / total_images) * 100
                    quality_score -= min(30, invalid_pct)  # Penalti maksimum 30 poin
                
                # Pastikan skor dalam range 0-100
                quality_score = max(0, min(100, quality_score))
                
                # Tambahkan ke hasil
                analysis_results['quality_scores'][split] = quality_score
                analysis_results['class_distribution'][split] = class_balance
                analysis_results['layer_distribution'][split] = layer_balance
                analysis_results['image_size_stats'][split] = image_sizes
                analysis_results['bbox_statistics'][split] = bbox_stats
                
                # Generate rekomendasi
                recommendations = self._generate_recommendations(
                    split=split,
                    quality_score=quality_score,
                    class_balance=class_balance,
                    layer_balance=layer_balance,
                    validation=validation
                )
                
                analysis_results['recommendations'].extend(recommendations)
                
                # Log hasil
                quality_category = "Baik" if quality_score >= 80 else "Sedang" if quality_score >= 60 else "Kurang"
                self.logger.info(
                    f"📊 Kualitas {split}: {quality_score:.1f}/100 ({quality_category})"
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error analisis {split}: {str(e)}")
                analysis_results[split] = {'error': str(e)}
        
        # Durasi analisis
        analysis_results['duration'] = time.time() - analysis_start
        
        self.logger.success(
            f"✨ Analisis kualitas dataset selesai dalam {analysis_results['duration']:.1f} detik"
        )
        
        # Tampilkan rekomendasi utama
        if analysis_results['recommendations']:
            self.logger.info("📋 Rekomendasi Utama:")
            for i, rec in enumerate(analysis_results['recommendations'][:5]):
                self.logger.info(f"   {i+1}. {rec}")
        
        return analysis_results
    
    def _generate_recommendations(
        self,
        split: str,
        quality_score: float,
        class_balance: Dict,
        layer_balance: Dict,
        validation: Dict
    ) -> List[str]:
        """
        Generate rekomendasi berdasarkan analisis kualitas.
        
        Args:
            split: Split dataset
            quality_score: Skor kualitas (0-100)
            class_balance: Statistik keseimbangan kelas
            layer_balance: Statistik keseimbangan layer
            validation: Hasil validasi dasar
            
        Returns:
            List rekomendasi
        """
        recommendations = []
        
        # Rekomendasi berdasarkan skor kualitas umum
        if quality_score < 60:
            recommendations.append(
                f"Kualitas dataset {split} perlu ditingkatkan secara signifikan (skor: {quality_score:.1f}/100)"
            )
        
        # Rekomendasi berdasarkan ketidakseimbangan kelas
        if class_balance.get('imbalance_score', 0) > 5:
            underrepresented = class_balance.get('underrepresented_classes', [])
            if underrepresented:
                recommendations.append(
                    f"Tambahkan lebih banyak data untuk kelas yang kurang terwakili: {', '.join(underrepresented[:3])}"
                )
            
            overrepresented = class_balance.get('overrepresented_classes', [])
            if overrepresented:
                recommendations.append(
                    f"Kurangi dominasi kelas: {', '.join(overrepresented[:3])}"
                )
        
        # Rekomendasi berdasarkan ketidakseimbangan layer
        if layer_balance.get('imbalance_score', 0) > 5:
            layer_percentages = layer_balance.get('layer_percentages', {})
            sorted_layers = sorted(layer_percentages.items(), key=lambda x: x[1])
            
            if sorted_layers and len(sorted_layers) > 1:
                lowest_layer, lowest_pct = sorted_layers[0]
                recommendations.append(
                    f"Tingkatkan representasi layer {lowest_layer} (hanya {lowest_pct:.1f}%)"
                )
        
        # Rekomendasi berdasarkan validasi dasar
        missing_labels = validation.get('missing_labels', 0)
        if missing_labels > 0:
            recommendations.append(
                f"Tambahkan label untuk {missing_labels} gambar tanpa label di {split}"
            )
        
        invalid_labels = validation.get('invalid_labels', 0)
        if invalid_labels > 0:
            recommendations.append(
                f"Perbaiki {invalid_labels} label tidak valid di {split}"
            )
        
        corrupt_images = validation.get('corrupt_images', 0)
        if corrupt_images > 0:
            recommendations.append(
                f"Ganti {corrupt_images} gambar rusak di {split}"
            )
        
        return recommendations
    
    def export_dataset_report(
        self,
        analysis_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export laporan analisis dataset ke file Markdown.
        
        Args:
            analysis_results: Hasil analisis dari analyze_dataset_quality()
            output_path: Path output (jika None, generate otomatis)
            
        Returns:
            Path file laporan
        """
        if output_path is None:
            timestamp = int(time.time())
            output_path = str(self.output_dir / f"dataset_report_{timestamp}.md")
            
        output_path = Path(output_path)
        
        try:
            # Generate isi laporan
            report_content = [
                "# Laporan Kualitas Dataset SmartCash",
                f"Tanggal: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Ringkasan Kualitas",
                "",
                "| Split | Skor Kualitas | Status |",
                "|-------|---------------|--------|",
            ]
            
            # Tambahkan skor per split
            for split, score in analysis_results.get('quality_scores', {}).items():
                status = "✅ Baik" if score >= 80 else "⚠️ Sedang" if score >= 60 else "❌ Kurang"
                report_content.append(f"| {split} | {score:.1f}/100 | {status} |")
                
            # Tambahkan rekomendasi
            report_content.extend([
                "",
                "## Rekomendasi Utama",
                ""
            ])
            
            for i, rec in enumerate(analysis_results.get('recommendations', [])[:10]):
                report_content.append(f"{i+1}. {rec}")
                
            # Detail per split
            for split in analysis_results.get('quality_scores', {}).keys():
                report_content.extend([
                    "",
                    f"## Detail Split: {split}",
                    "",
                    "### Distribusi Kelas",
                    ""
                ])
                
                # Tambahkan distribusi kelas
                class_dist = analysis_results.get('class_distribution', {}).get(split, {})
                if class_dist:
                    class_percentages = class_dist.get('class_percentages', {})
                    imbalance = class_dist.get('imbalance_score', 0)
                    
                    report_content.append(f"Skor ketidakseimbangan: {imbalance:.1f}/10")
                    report_content.append("")
                    report_content.append("| Kelas | Persentase |")
                    report_content.append("|-------|------------|")
                    
                    for cls, pct in list(sorted(class_percentages.items(), key=lambda x: x[1], reverse=True))[:10]:
                        report_content.append(f"| {cls} | {pct:.1f}% |")
                        
                # Tambahkan distribusi layer
                report_content.extend([
                    "",
                    "### Distribusi Layer",
                    ""
                ])
                
                layer_dist = analysis_results.get('layer_distribution', {}).get(split, {})
                if layer_dist:
                    layer_percentages = layer_dist.get('layer_percentages', {})
                    imbalance = layer_dist.get('imbalance_score', 0)
                    
                    report_content.append(f"Skor ketidakseimbangan: {imbalance:.1f}/10")
                    report_content.append("")
                    report_content.append("| Layer | Persentase |")
                    report_content.append("|-------|------------|")
                    
                    for layer, pct in sorted(layer_percentages.items(), key=lambda x: x[1], reverse=True):
                        report_content.append(f"| {layer} | {pct:.1f}% |")
            
            # Tulis ke file
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_content))
                
            self.logger.success(f"✅ Laporan dataset diekspor ke {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor laporan: {str(e)}")
            return ""
    
    def clean_cache(self) -> bool:
        """
        Bersihkan cache preprocessing.
        
        Returns:
            Boolean sukses/gagal
        """
        try:
            cache_stats = self.cache.get_stats()
            self.logger.info(
                f"🧹 Membersihkan cache preprocessing:\n"
                f"   • Ukuran: {cache_stats['cache_size_mb']:.1f} MB\n"
                f"   • File: {cache_stats['file_count']}"
            )
            
            self.cache.clear()
            
            self.logger.success("✅ Cache berhasil dibersihkan")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan cache: {str(e)}")
            return False