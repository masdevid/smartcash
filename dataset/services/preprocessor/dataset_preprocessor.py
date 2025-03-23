"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Layanan preprocessing dataset dengan pelaporan progres yang dioptimalkan dan refactor untuk DRY
"""

import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import cv2
import numpy as np
from tqdm.auto import tqdm

from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.services.preprocessor.pipeline import PreprocessingPipeline
from smartcash.dataset.services.preprocessor.storage import PreprocessedStorage
from smartcash.components.observer import notify, EventTopics

# Import utils yang telah dipindahkan
from smartcash.dataset.utils.preprocessor_utils import (
    extract_class_from_label,
    get_source_dir,
    update_progress,
    resolve_splits,
    preprocess_single_image
)

class DatasetPreprocessor:
    """Layanan preprocessing dataset dengan pelaporan progres yang dioptimalkan."""
    
    def __init__(self, config, logger=None):
        """
        Inisialisasi preprocessor.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging (opsional)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._progress_callback = None
        
        # Inisialisasi pipeline preprocessing
        self.pipeline = PreprocessingPipeline(config, self.logger)
        
        # Setup direktori output
        output_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        self.storage = PreprocessedStorage(output_dir, self.logger)
        
        # Prefix untuk penamaan file
        self.file_prefix = config.get('preprocessing', {}).get('file_prefix', 'rp')
        
    def register_progress_callback(self, callback: Callable) -> None:
        """
        Register progress callback untuk tracking progress.
        
        Args:
            callback: Progress callback function
        """
        self._progress_callback = callback
    
    def preprocess_dataset(self, split: str = None, force_reprocess: bool = False, show_progress: bool = True, **kwargs) -> Dict[str, Any]:
        """Preprocess dataset dan simpan hasil untuk penggunaan berikutnya."""
        try:
            try: notify(event_type=EventTopics.PREPROCESSING_START, sender="dataset_preprocessor", message=f"Memulai preprocessing dataset {split or 'semua split'}") 
            except Exception: pass
            
            if not (splits_to_process := resolve_splits(split)): return {"error": "Tidak ada split valid"}
            start_time, results = time.time(), {}
            
            # Definisi langkah-langkah utama preprocessing dengan step metadata
            steps = [
                {"name": "Persiapan dataset", "weight": 0.1},        # 10% dari total progress
                {"name": "Memproses dataset", "weight": 0.8},        # 80% dari total progress
                {"name": "Finalisasi hasil", "weight": 0.1}          # 10% dari total progress
            ]
            
            # PERBAIKAN: Hitung total gambar di semua split terlebih dahulu
            total_images_by_split = {}
            total_images_all = 0
            for current_split in splits_to_process:
                images_dir = Path(self._get_source_dir(current_split)) / 'images'
                if images_dir.exists():
                    split_images = len(list(images_dir.glob('*.*')))
                    total_images_by_split[current_split] = split_images
                    total_images_all += split_images
            
            # Tambahkan buffer untuk langkah persiapan dan finalisasi
            total_progress = total_images_all + 2  # +1 persiapan, +1 finalisasi
            
            # Step 1: Persiapan dataset - 10% dari total progress
            update_progress(
                self._progress_callback, 1, total_progress, 
                f"Persiapan preprocessing dataset ({total_images_all} gambar di {len(splits_to_process)} split)...", 
                status="info", step=0, split_step="0/1", 
                total_files_all=total_progress
            )
            
            # Step 2: Proses setiap split - 80% dari total progress
            progress_so_far = 1  # Mulai dari 1 (setelah persiapan)
            
            for i, current_split in enumerate(splits_to_process):
                # PERBAIKAN: Update progress untuk mulai memproses split ini dengan metadata lebih lengkap
                update_progress(
                    self._progress_callback, progress_so_far, total_progress, 
                    f"Memulai preprocessing split {current_split} ({total_images_by_split.get(current_split, 0)} gambar)...", 
                    status="info", step=1, split=current_split, 
                    split_step=f"{i+1}/{len(splits_to_process)}",
                    total_files_all=total_progress
                )
                
                # Proses split dan dapatkan hasil dengan parameter yang ditingkatkan
                split_result = self._preprocess_split(
                    current_split, force_reprocess, show_progress, 
                    progress_start=progress_so_far, 
                    total_progress=total_progress, 
                    total_files_all=total_progress,
                    split_step=f"{i+1}/{len(splits_to_process)}",
                    **kwargs
                )
                
                results[current_split] = split_result
                
                # Update progress setelah split selesai
                images_processed = split_result.get('processed', 0) + split_result.get('skipped', 0)
                progress_so_far += images_processed
                
                # PERBAIKAN: Memberikan pesan ringkas tentang hasil split dengan informasi lebih lengkap
                split_msg = f"Preprocessing Split {current_split} ({i+1}/{len(splits_to_process)}) selesai: {split_result.get('processed', 0)} diproses, {split_result.get('skipped', 0)} dilewati"
                update_progress(
                    self._progress_callback, progress_so_far, total_progress, split_msg, 
                    status="info", step=1, split=current_split, 
                    split_step=f"{i+1}/{len(splits_to_process)}",
                    total_files_all=total_progress
                )
            
            # Step 3: Finalisasi - 10% dari total progress
            update_progress(
                self._progress_callback, total_progress - 1, total_progress, 
                f"Finalisasi hasil preprocessing...", status="info",
                step=2, split_step="1/1",
                total_files_all=total_progress
            )
            
            # Hitung total statistik
            total_images, total_skipped, total_failed = [sum(r.get(k, 0) for r in results.values()) for k in ('processed', 'skipped', 'failed')]
            
            # Generate hasil
            total_result = {
                'total_images': total_images, 'total_skipped': total_skipped, 'total_failed': total_failed,
                'split_stats': {s: {'images': r.get('processed', 0), 'labels': r.get('processed', 0), 
                                'complete': r.get('processed', 0) > 0 and r.get('failed', 0) == 0} for s, r in results.items()},
                'processing_time': time.time() - start_time, 'success': total_failed == 0,
                'output_dir': str(self.storage.output_dir),
                'image_size': self.config.get('preprocessing', {}).get('img_size', [640, 640])
            }
            
            # Update progress final (100%)
            update_progress(
                self._progress_callback, total_progress, total_progress, 
                f"Preprocessing selesai: {total_images} gambar berhasil diproses dalam {total_result['processing_time']:.1f} detik", 
                status="success", step=2, split_step="1/1",
                total_files_all=total_progress
            )
            
            try: notify(event_type=EventTopics.PREPROCESSING_END, sender="dataset_preprocessor", 
                       message=f"Preprocessing selesai: {total_images} gambar berhasil diproses", 
                       duration=total_result['processing_time'])
            except Exception: pass
                
            return total_result
                
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan preprocessing: {str(e)}")
            try: notify(event_type=EventTopics.PREPROCESSING_ERROR, sender="dataset_preprocessor", message=f"Error preprocessing: {str(e)}")
            except Exception: pass
            raise DatasetProcessingError(f"Gagal melakukan preprocessing: {str(e)}")
    
    def _get_source_dir(self, split: str) -> str:
        """Proxy method ke utils function untuk backward compatibility."""
        return get_source_dir(split, self.config)
    
    def _preprocess_split(self, split: str, force_reprocess: bool, show_progress: bool, 
                         progress_start: int = 0, total_progress: int = 100, 
                         split_step: str = "", total_files_all: int = 0, **kwargs) -> Dict[str, Any]:
        """Preprocess satu split dataset dengan tracking progress yang lebih detail."""
        try:
            # Cek apakah sudah dipreprocess dan tidak perlu diproses ulang
            if not force_reprocess and self.is_preprocessed(split):
                self.logger.info(f"✅ Split '{split}' sudah dipreprocess sebelumnya")
                processed = len(list((images_path := self.storage.get_split_path(split) / 'images').glob('*.*'))) if images_path.exists() else 0
                
                # Update progress untuk menandai split sudah selesai diproses
                update_progress(
                    self._progress_callback, progress_start + processed, total_progress,
                    f"Split {split} sudah dipreprocess sebelumnya ({processed} gambar)", status="info",
                    step=1, split=split, split_step=split_step,
                    total_files_all=total_files_all
                )
                                    
                return {'processed': processed, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Dapatkan path source dan validasi direktori
            source_dir, images_dir, labels_dir = self._get_source_dir(split), Path(self._get_source_dir(split)) / 'images', Path(self._get_source_dir(split)) / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                error_message = f"❌ Direktori gambar atau label tidak ditemukan: {images_dir} atau {labels_dir}"
                self.logger.error(error_message)
                update_progress(
                    self._progress_callback, progress_start, total_progress, error_message, 
                    status="error", step=1, split=split, split_step=split_step,
                    total_files_all=total_files_all
                )
                return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': error_message}
                
            # Setup direktori target
            target_dir, target_images_dir, target_labels_dir = self.storage.get_split_path(split), self.storage.get_split_path(split) / 'images', self.storage.get_split_path(split) / 'labels'
            target_images_dir.mkdir(parents=True, exist_ok=True); target_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Mulai preprocessing
            self.logger.info(f"🔄 Memulai preprocessing untuk split '{split}'")
            
            # Dapatkan daftar file dan validasi
            image_files, num_files = sorted(list(images_dir.glob('*.*'))), len(image_files := sorted(list(images_dir.glob('*.*'))))
            
            if num_files == 0:
                self.logger.warning(f"⚠️ Tidak ada file gambar yang ditemukan di {images_dir}")
                update_progress(
                    self._progress_callback, progress_start, total_progress, 
                    f"Tidak ada file gambar ditemukan di {images_dir}", 
                    status="warning", step=1, split=split, split_step=split_step,
                    total_files_all=total_files_all
                )
                return {'processed': 0, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Setup progress tracking dan statistik
            progress = tqdm(total=num_files, desc=f"Preprocessing {split}", disable=not show_progress)
            stats = {'processed': 0, 'skipped': 0, 'failed': 0, 'start_time': time.time()}
            
            # Ekstrak opsi preprocessing
            preprocessing_options = {
                'img_size': kwargs.get('img_size', self.config.get('preprocessing', {}).get('img_size', [640, 640])),
                'normalize': kwargs.get('normalize', self.config.get('preprocessing', {}).get('normalization', {}).get('enabled', True)),
                'preserve_aspect_ratio': kwargs.get('preserve_aspect_ratio', self.config.get('preprocessing', {}).get('normalization', {}).get('preserve_aspect_ratio', True))
            }
            
            # PERBAIKAN: Tambahkan notifikasi awal saat mulai preprocessing split
            update_progress(
                self._progress_callback, progress_start, total_progress,
                f"Memulai preprocessing split {split} ({num_files} gambar)...",
                status="info", step=1, split=split, split_step=split_step, 
                total_files_all=total_files_all,
                current_progress=0, current_total=num_files
            )
            
            # Proses setiap gambar
            for i, img_path in enumerate(image_files):
                try:
                    # PERBAIKAN: Update progress bar dengan informasi lebih lengkap dan konsisten
                    current_processed = i + 1
                    overall_processed = progress_start + current_processed
                    
                    # Update progress setiap 10% atau pada file terakhir
                    if current_processed == 1 or current_processed == num_files or current_processed % max(1, num_files // 10) == 0:
                        progress_msg = f"Preprocessing split {split}: {current_processed}/{num_files} gambar"
                        
                        update_progress(
                            self._progress_callback, overall_processed, total_progress, progress_msg, 
                            status='info', current_progress=current_processed,
                            current_total=num_files, split=split, split_step=split_step,
                            step=1, total_files_all=total_files_all
                        )
                    
                    # Proses gambar dan update statistic menggunakan fungsi dari utils
                    result = preprocess_single_image(
                        img_path, labels_dir, target_images_dir, target_labels_dir,
                        self.pipeline, self.storage, self.file_prefix,
                        preprocessing_options, self.logger
                    )
                    stats['processed' if result else 'skipped'] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Gagal memproses {img_path.name}: {str(e)}"); stats['failed'] += 1
                
                progress.update(1)
            
            progress.close()
            
            # Notifikasi finalisasi dan penyimpanan
            update_progress(
                self._progress_callback, progress_start + num_files, total_progress, 
                f"Menyimpan metadata dan finalisasi split {split}", status="info", 
                step=1, split=split, split_step=split_step,
                total_files_all=total_files_all
            )
            
            # Update statistik dan simpan metadata
            stats.update({'end_time': time.time(), 'duration': time.time() - stats['start_time'], 'total': num_files, 'success': stats['failed'] == 0})
            self.storage.update_stats(split, stats)
            
            # PERBAIKAN: Tampilkan pesan selesai preprocessing split
            summary_message = f"✅ Preprocessing split {split} selesai: {stats['processed']} berhasil, {stats['skipped']} dilewati, {stats['failed']} gagal, dalam {stats['duration']:.2f} detik"
            self.logger.info(summary_message)
            
            # Final update untuk split
            update_progress(
                self._progress_callback, progress_start + num_files, total_progress, 
                f"Preprocessing split {split} selesai", status="success", 
                step=1, split=split, split_step=split_step,
                total_files_all=total_files_all
            )
            
            return stats
                
        except Exception as e:
            self.logger.error(f"❌ Gagal preprocessing split {split}: {str(e)}")
            return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': str(e)}
    
    def clean_preprocessed(self, split: Optional[str] = None) -> None:
        """
        Bersihkan hasil preprocessing untuk split tertentu atau semua split.
        
        Args:
            split: Nama split yang akan dibersihkan, atau None untuk semua
        """
        if split:
            self.logger.info(f"🧹 Membersihkan hasil preprocessing untuk split '{split}'")
            self.storage.clean_storage(split)
        else:
            self.logger.info(f"🧹 Membersihkan semua hasil preprocessing")
            for split_name in ['train', 'valid', 'test']:
                self.storage.clean_storage(split_name)
    
    def get_preprocessed_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik hasil preprocessing.
        
        Returns:
            Dictionary berisi statistik preprocessing
        """
        stats = {}
        for split_name in ['train', 'valid', 'test']:
            split_stats = self.storage.get_stats(split_name)
            if split_stats:
                stats[split_name] = split_stats
                
        return stats
    
    def is_preprocessed(self, split: str) -> bool:
        """
        Cek apakah split sudah dipreprocess.
        
        Args:
            split: Nama split
            
        Returns:
            Boolean menunjukkan apakah split sudah dipreprocess
        """
        split_path = self.storage.get_split_path(split)
        return split_path.exists() and len(list(split_path.glob('**/*.jpg'))) > 0

# Class alias untuk backward compatibility
DatasetPreprocessorService = DatasetPreprocessor