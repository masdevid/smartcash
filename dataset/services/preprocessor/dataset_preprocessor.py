"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Layanan preprocessing dataset dengan one-liner dan pelaporan progres yang dioptimalkan 
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

# Import utils yang dioptimalkan
from smartcash.dataset.utils.label_utils import extract_class_from_label
from smartcash.dataset.utils.move_utils import get_source_dir, resolve_splits, calculate_total_images
from smartcash.dataset.utils.progress_utils import update_progress
from smartcash.dataset.utils.preprocessing_image_utils import preprocess_single_image

class DatasetPreprocessor:
    """Layanan preprocessing dataset dengan pelaporan progres yang dioptimalkan."""
    
    def __init__(self, config, logger=None):
        """Inisialisasi preprocessor dengan parameter standard."""
        self.config, self.logger = config, logger or logging.getLogger(__name__)
        self._progress_callback = None
        
        # Inisialisasi components dengan one-liner
        self.pipeline = PreprocessingPipeline(config, self.logger)
        self.storage = PreprocessedStorage(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'), self.logger)
        self.file_prefix = config.get('preprocessing', {}).get('file_prefix', 'rp')
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk tracking progress."""
        self._progress_callback = callback
    
    def preprocess_dataset(self, split: str = None, force_reprocess: bool = False, show_progress: bool = True, **kwargs) -> Dict[str, Any]:
        """Preprocess dataset dan simpan hasil dengan pelaporan progres yang dioptimalkan."""
        try:
            try: notify(event_type=EventTopics.PREPROCESSING_START, sender="dataset_preprocessor", message=f"Memulai preprocessing dataset {split or 'semua split'}") 
            except Exception: pass
            
            # Validasi dan inisialisasi
            if not (splits_to_process := resolve_splits(split)): return {"error": "Tidak ada split valid"}
            start_time, results = time.time(), {}
            
            # Definisi langkah preprocessing dengan metadata
            steps = [
                {"name": "Persiapan dataset", "weight": 0.1},
                {"name": "Memproses dataset", "weight": 0.8},
                {"name": "Finalisasi hasil", "weight": 0.1}
            ]
            
            # Hitung total gambar dengan helper function
            total_images_by_split = calculate_total_images(splits_to_process, self.config)
            total_images_all = total_images_by_split['all']
            total_progress = total_images_all + 2  # +2 buffer untuk persiapan dan finalisasi
            
            # Step 1: Persiapan dataset (10%)
            update_progress(
                self._progress_callback, 1, total_progress, 
                f"Persiapan preprocessing dataset ({total_images_all} gambar di {len(splits_to_process)} split)...", 
                status="info", step=0, split_step="Persiapan (1/3)", 
                total_files_all=total_progress
            )
            
            # Step 2: Proses setiap split (80%)
            progress_so_far = 1  # Setelah persiapan
            
            for i, current_split in enumerate(splits_to_process):
                # Dapatkan jumlah file dari hasil perhitungan awal
                split_image_count = total_images_by_split.get(current_split, 0)
                
                # Update progress awal split
                split_step_text = f"Split {i+1}/{len(splits_to_process)}"
                update_progress(
                    self._progress_callback, progress_so_far, total_progress, 
                    f"Memulai preprocessing split {current_split} ({split_image_count} gambar)...", 
                    status="info", step=1, split=current_split, 
                    split_step=split_step_text,
                    total_files_all=total_progress,
                    current_progress=0,
                    current_total=split_image_count
                )
                
                # Proses split dengan parameter yang dioptimalkan
                split_result = self._preprocess_split(
                    current_split, force_reprocess, show_progress, 
                    progress_start=progress_so_far, 
                    total_progress=total_progress, 
                    total_files_all=total_progress,
                    split_step=split_step_text,
                    **kwargs
                )
                
                results[current_split] = split_result
                
                # Update progress setelah split selesai
                progress_so_far += split_result.get('processed', 0) + split_result.get('skipped', 0)
                
                # Tampilkan ringkasan hasil split
                update_progress(
                    self._progress_callback, progress_so_far, total_progress, 
                    f"Preprocessing Split {current_split} ({i+1}/{len(splits_to_process)}) selesai: {split_result.get('processed', 0)} diproses, {split_result.get('skipped', 0)} dilewati", 
                    status="info", step=1, split=current_split, 
                    split_step=split_step_text,
                    total_files_all=total_progress,
                    current_progress=split_image_count,
                    current_total=split_image_count
                )
            
            # Step 3: Finalisasi (10%)
            update_progress(
                self._progress_callback, total_progress - 1, total_progress, 
                f"Finalisasi hasil preprocessing...", status="info",
                step=2, split_step="Finalisasi (3/3)",
                total_files_all=total_progress
            )
            
            # Hitung statistik agregat dengan list comprehension
            total_images, total_skipped, total_failed = [sum(r.get(k, 0) for r in results.values()) for k in ('processed', 'skipped', 'failed')]
            
            # Siapkan hasil dengan dictionary comprehension
            total_result = {
                'total_images': total_images, 
                'total_skipped': total_skipped, 
                'total_failed': total_failed,
                'split_stats': {s: {'images': r.get('processed', 0), 
                                   'labels': r.get('processed', 0), 
                                   'complete': r.get('processed', 0) > 0 and r.get('failed', 0) == 0} 
                               for s, r in results.items()},
                'processing_time': time.time() - start_time, 
                'success': total_failed == 0,
                'output_dir': str(self.storage.output_dir),
                'image_size': self.config.get('preprocessing', {}).get('img_size', [640, 640])
            }
            
            # Update progress final (100%)
            update_progress(
                self._progress_callback, total_progress, total_progress, 
                f"Preprocessing selesai: {total_images} gambar berhasil diproses dalam {total_result['processing_time']:.1f} detik", 
                status="success", step=2, split_step="Finalisasi (3/3)",
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
        """Dapatkan direktori source untuk split."""
        return get_source_dir(split, self.config)
    
    def _preprocess_split(self, split: str, force_reprocess: bool, show_progress: bool, 
                         progress_start: int = 0, total_progress: int = 100, 
                         split_step: str = "", total_files_all: int = 0, **kwargs) -> Dict[str, Any]:
        """Preprocess satu split dataset dengan tracking progress yang dioptimalkan."""
        try:
            # Cek apakah sudah dipreprocess dan tidak perlu diproses ulang
            if not force_reprocess and self.is_preprocessed(split):
                processed = len(list((images_path := self.storage.get_split_path(split) / 'images').glob('*.*'))) if images_path.exists() else 0
                
                # Update progress untuk split yang sudah diproses
                update_progress(
                    self._progress_callback, progress_start + processed, total_progress,
                    f"Split {split} sudah dipreprocess sebelumnya ({processed} gambar)", status="info",
                    step=1, split=split, split_step=split_step,
                    total_files_all=total_files_all,
                    current_progress=processed,
                    current_total=processed
                )
                                    
                return {'processed': processed, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Setup dan validasi direktori
            source_dir = self._get_source_dir(split)
            images_dir, labels_dir = Path(source_dir) / 'images', Path(source_dir) / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                error_message = f"❌ Direktori gambar atau label tidak ditemukan: {images_dir} atau {labels_dir}"
                self.logger.error(error_message)
                update_progress(
                    self._progress_callback, progress_start, total_progress, error_message, 
                    status="error", step=1, split=split, split_step=split_step,
                    total_files_all=total_files_all
                )
                return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': error_message}
                
            # Setup direktori target dengan one-liner
            target_dir = self.storage.get_split_path(split)
            target_images_dir, target_labels_dir = target_dir / 'images', target_dir / 'labels'
            [d.mkdir(parents=True, exist_ok=True) for d in [target_images_dir, target_labels_dir]]
            
            # Mulai preprocessing
            self.logger.info(f"🔄 Memulai preprocessing untuk split '{split}'")
            
            # Dapatkan file dan validasi
            image_files = sorted(list(images_dir.glob('*.*')))
            num_files = len(image_files)
            
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
            
            # Ekstrak opsi preprocessing dengan dictionary unpacking
            preprocessing_options = {
                'img_size': kwargs.get('img_size', self.config.get('preprocessing', {}).get('img_size', [640, 640])),
                'normalize': kwargs.get('normalize', self.config.get('preprocessing', {}).get('normalization', {}).get('enabled', True)),
                'preserve_aspect_ratio': kwargs.get('preserve_aspect_ratio', self.config.get('preprocessing', {}).get('normalization', {}).get('preserve_aspect_ratio', True))
            }
            
            # Notifikasi awal preprocessing split
            update_progress(
                self._progress_callback, progress_start, total_progress,
                f"Memulai preprocessing split {split} ({num_files} gambar)...",
                status="info", step=1, split=split, split_step=split_step, 
                total_files_all=total_files_all,
                current_progress=0, current_total=num_files
            )
            
            # Proses setiap gambar dengan progress tracking yang dioptimalkan
            for i, img_path in enumerate(image_files):
                try:
                    # Update progress dengan throttling
                    current_processed = i + 1
                    split_proportion = 1.0 / len(resolve_splits(None))
                    overall_progress = progress_start + current_processed
                    
                    # Progress updates with throttling
                    if current_processed == 1 or current_processed == num_files or current_processed % max(1, num_files // 10) == 0:
                        update_progress(
                            self._progress_callback, 
                            min(overall_progress, total_progress),
                            total_progress, f"Preprocessing split {split}: {current_processed}/{num_files} gambar", 
                            status='info', current_progress=current_processed,
                            current_total=num_files, split=split, split_step=split_step,
                            step=1, total_files_all=total_files_all
                        )
                    
                    # Proses gambar dan update statistik
                    result = preprocess_single_image(
                        img_path, labels_dir, target_images_dir, target_labels_dir,
                        self.pipeline, self.storage, self.file_prefix,
                        preprocessing_options, self.logger
                    )
                    stats['processed' if result else 'skipped'] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Gagal memproses {img_path.name}: {str(e)}")
                    stats['failed'] += 1
                
                progress.update(1)
            
            progress.close()
            
            # Finalisasi dan simpan metadata
            update_progress(
                self._progress_callback, progress_start + num_files, total_progress, 
                f"Menyimpan metadata dan finalisasi split {split}", status="info", 
                step=1, split=split, split_step=split_step,
                total_files_all=total_files_all,
                current_progress=num_files,
                current_total=num_files
            )
            
            # Update statistik dan simpan
            stats.update({
                'end_time': time.time(), 
                'duration': time.time() - stats['start_time'], 
                'total': num_files, 
                'success': stats['failed'] == 0
            })
            
            self.storage.update_stats(split, stats)
            
            # Tampilkan ringkasan hasil preprocessing split
            self.logger.info(f"✅ Preprocessing split {split} selesai: {stats['processed']} berhasil, {stats['skipped']} dilewati, {stats['failed']} gagal, dalam {stats['duration']:.2f} detik")
            
            # Final update untuk split
            update_progress(
                self._progress_callback, progress_start + num_files, total_progress, 
                f"Preprocessing split {split} selesai", status="success", 
                step=1, split=split, split_step=split_step,
                total_files_all=total_files_all,
                current_progress=num_files,
                current_total=num_files
            )
            
            return stats
                
        except Exception as e:
            self.logger.error(f"❌ Gagal preprocessing split {split}: {str(e)}")
            return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': str(e)}
    
    def clean_preprocessed(self, split: Optional[str] = None) -> None:
        """Bersihkan hasil preprocessing untuk split tertentu atau semua split."""
        if split:
            self.logger.info(f"🧹 Membersihkan hasil preprocessing untuk split '{split}'")
            self.storage.clean_storage(split)
        else:
            self.logger.info(f"🧹 Membersihkan semua hasil preprocessing")
            [self.storage.clean_storage(split_name) for split_name in ['train', 'valid', 'test']]
    
    def get_preprocessed_stats(self) -> Dict[str, Any]:
        """Dapatkan statistik hasil preprocessing dengan dictionary comprehension."""
        return {split_name: self.storage.get_stats(split_name) 
                for split_name in ['train', 'valid', 'test'] 
                if self.storage.get_stats(split_name)}
    
    def is_preprocessed(self, split: str) -> bool:
        """Cek apakah split sudah dipreprocess."""
        split_path = self.storage.get_split_path(split)
        return split_path.exists() and len(list(split_path.glob('**/*.jpg'))) > 0

# Class alias untuk backward compatibility
DatasetPreprocessorService = DatasetPreprocessor