"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Dataset preprocessor dengan UI progress notifications dan suppressed console logging
"""

import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import cv2
import numpy as np

from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.services.preprocessor.pipeline import PreprocessingPipeline
from smartcash.dataset.services.preprocessor.storage import PreprocessedStorage

# Import utils yang dioptimalkan
from smartcash.dataset.utils.move_utils import get_source_dir, resolve_splits, calculate_total_images
from smartcash.dataset.utils.preprocessing_image_utils import preprocess_single_image
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_IMG_SIZE

class DatasetPreprocessor:
    """Dataset preprocessor dengan UI progress notifications dan suppressed console logging."""
    
    def __init__(self, config, logger=None):
        """Inisialisasi preprocessor dengan suppressed console logging."""
        self.config, self.logger = config, logger or logging.getLogger(__name__)
        self._progress_callback = None
        
        # Setup suppressed logging
        self._setup_suppressed_logging()
        
        # Inisialisasi components dengan one-liner
        self.pipeline = PreprocessingPipeline(config, self.logger)
        self.storage = PreprocessedStorage(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'), self.logger)
        self.file_prefix = config.get('preprocessing', {}).get('file_prefix', 'rp')
    
    def _setup_suppressed_logging(self):
        """Setup logging yang disuppress untuk prevent console output."""
        # Suppress tqdm dan progress bar console output
        import sys
        from io import StringIO
        
        # Store original untuk kemungkinan restore
        if not hasattr(self, '_original_stdout'):
            self._original_stdout = sys.stdout
        
        # Suppress semua print statements dari preprocessing
        self._suppress_console = True
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk UI notifications."""
        self._progress_callback = callback
    
    def _notify_progress(self, **kwargs):
        """Notify progress ke UI tanpa console output."""
        if self._progress_callback and callable(self._progress_callback):
            try:
                # Set default values
                params = {
                    'progress': kwargs.get('progress', 0),
                    'total': kwargs.get('total', 100),
                    'message': kwargs.get('message', 'Processing...'),
                    'status': kwargs.get('status', 'info'),
                    'step': kwargs.get('step', 0),
                    'split_step': kwargs.get('split_step', ''),
                    'current_progress': kwargs.get('current_progress', 0),
                    'current_total': kwargs.get('current_total', 0),
                    'split': kwargs.get('split', ''),
                    'total_files_all': kwargs.get('total_files_all', 0)
                }
                
                # Jalankan callback tanpa error handling yang verbose
                self._progress_callback(**params)
            except Exception:
                # Silent fail untuk prevent console logs
                pass
    
    def preprocess_dataset(self, split: str = None, force_reprocess: bool = False, show_progress: bool = False, **kwargs) -> Dict[str, Any]:
        """Preprocess dataset dengan UI progress notifications."""
        try:
            # Notify start
            self._notify_progress(
                progress=0, total=100,
                message=f"Memulai preprocessing dataset {split or 'semua split'}",
                status="info", step=0, split_step="Persiapan"
            )
            
            # Validasi dan inisialisasi
            if not (splits_to_process := resolve_splits(split)): 
                return {"error": "Tidak ada split valid"}
            
            start_time, results = time.time(), {}
            
            # Hitung total gambar untuk progress tracking
            total_images_by_split = calculate_total_images(splits_to_process, self.config)
            total_images_all = total_images_by_split['all']
            
            # Step 1: Persiapan (5%)
            self._notify_progress(
                progress=5, total=100,
                message=f"Persiapan preprocessing {total_images_all} gambar di {len(splits_to_process)} split",
                status="info", step=1, split_step="Persiapan (1/3)",
                total_files_all=total_images_all
            )
            
            # Step 2: Proses setiap split (85%)
            progress_base = 5
            progress_per_split = 85 / len(splits_to_process)
            
            for i, current_split in enumerate(splits_to_process):
                split_image_count = total_images_by_split.get(current_split, 0)
                split_step_text = f"Split {current_split} ({i+1}/{len(splits_to_process)})"
                
                # Notify start split
                current_progress = progress_base + (i * progress_per_split)
                self._notify_progress(
                    progress=int(current_progress), total=100,
                    message=f"Memulai preprocessing split {current_split}",
                    status="info", step=2, split_step=split_step_text,
                    split=current_split, current_progress=0, current_total=split_image_count,
                    total_files_all=total_images_all
                )
                
                # Proses split dengan progress callback
                split_result = self._preprocess_split(
                    current_split, force_reprocess, 
                    progress_base=current_progress,
                    progress_range=progress_per_split,
                    split_step=split_step_text,
                    total_files_all=total_images_all,
                    **kwargs
                )
                
                results[current_split] = split_result
                
                # Notify split complete
                final_progress = progress_base + ((i + 1) * progress_per_split)
                self._notify_progress(
                    progress=int(final_progress), total=100,
                    message=f"Split {current_split} selesai: {split_result.get('processed', 0)} diproses",
                    status="success", step=2, split_step=split_step_text,
                    split=current_split, current_progress=split_image_count, current_total=split_image_count,
                    total_files_all=total_images_all
                )
            
            # Step 3: Finalisasi (10%)
            self._notify_progress(
                progress=90, total=100,
                message="Finalisasi hasil preprocessing...",
                status="info", step=3, split_step="Finalisasi (3/3)",
                total_files_all=total_images_all
            )
            
            # Hitung statistik agregat
            total_images, total_skipped, total_failed = [
                sum(r.get(k, 0) for r in results.values()) 
                for k in ('processed', 'skipped', 'failed')
            ]
            
            # Siapkan hasil
            total_result = {
                'total_images': total_images, 
                'total_skipped': total_skipped, 
                'total_failed': total_failed,
                'split_stats': {
                    s: {
                        'images': r.get('processed', 0), 
                        'labels': r.get('processed', 0), 
                        'complete': r.get('processed', 0) > 0 and r.get('failed', 0) == 0
                    } 
                    for s, r in results.items()
                },
                'processing_time': time.time() - start_time, 
                'success': total_failed == 0,
                'output_dir': str(self.storage.output_dir),
                'image_size': self.config.get('preprocessing', {}).get('img_size', DEFAULT_IMG_SIZE)
            }
            
            # Final notification
            self._notify_progress(
                progress=100, total=100,
                message=f"Preprocessing selesai: {total_images} gambar dalam {total_result['processing_time']:.1f} detik",
                status="success", step=3, split_step="Selesai",
                total_files_all=total_images_all
            )
                
            return total_result
                
        except Exception as e:
            self._notify_progress(
                progress=0, total=100,
                message=f"Error preprocessing: {str(e)}",
                status="error", step=0, split_step="Error"
            )
            raise DatasetProcessingError(f"Gagal melakukan preprocessing: {str(e)}")
    
    def _get_source_dir(self, split: str) -> str:
        """Dapatkan direktori source untuk split."""
        return get_source_dir(split, self.config)
    
    def _preprocess_split(self, split: str, force_reprocess: bool, 
                         progress_base: float = 0, progress_range: float = 100,
                         split_step: str = "", total_files_all: int = 0, **kwargs) -> Dict[str, Any]:
        """Preprocess satu split dengan UI progress notifications."""
        try:
            # Cek apakah sudah dipreprocess
            if not force_reprocess and self.is_preprocessed(split):
                processed = len(list((images_path := self.storage.get_split_path(split) / 'images').glob('*.*'))) if images_path.exists() else 0
                
                self._notify_progress(
                    progress=int(progress_base + progress_range), total=100,
                    message=f"Split {split} sudah dipreprocess ({processed} gambar)",
                    status="info", step=2, split_step=split_step,
                    split=split, current_progress=processed, current_total=processed,
                    total_files_all=total_files_all
                )                    
                return {'processed': processed, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Setup dan validasi direktori
            source_dir = self._get_source_dir(split)
            images_dir, labels_dir = Path(source_dir) / 'images', Path(source_dir) / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                error_message = f"Direktori tidak ditemukan: {images_dir} atau {labels_dir}"
                self._notify_progress(
                    progress=int(progress_base), total=100,
                    message=error_message, status="error",
                    step=2, split_step=split_step, split=split,
                    total_files_all=total_files_all
                )
                return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': error_message}
                
            # Setup direktori target
            target_dir = self.storage.get_split_path(split)
            target_images_dir, target_labels_dir = target_dir / 'images', target_dir / 'labels'
            [d.mkdir(parents=True, exist_ok=True) for d in [target_images_dir, target_labels_dir]]
            
            # Dapatkan file dan validasi
            image_files = sorted(list(images_dir.glob('*.*')))
            num_files = len(image_files)
            
            if num_files == 0:
                self._notify_progress(
                    progress=int(progress_base), total=100,
                    message=f"Tidak ada file gambar di {images_dir}",
                    status="warning", step=2, split_step=split_step, split=split,
                    total_files_all=total_files_all
                )
                return {'processed': 0, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Setup statistik
            stats = {'processed': 0, 'skipped': 0, 'failed': 0, 'start_time': time.time()}
            
            # Ekstrak opsi preprocessing
            preprocessing_options = {
                'img_size': kwargs.get('img_size', self.config.get('preprocessing', {}).get('img_size', DEFAULT_IMG_SIZE)),
                'normalize': kwargs.get('normalize', self.config.get('preprocessing', {}).get('normalization', {}).get('enabled', True)),
                'preserve_aspect_ratio': kwargs.get('preserve_aspect_ratio', self.config.get('preprocessing', {}).get('normalization', {}).get('preserve_aspect_ratio', True))
            }
            
            # Proses setiap gambar dengan progress tracking
            for i, img_path in enumerate(image_files):
                try:
                    current_processed = i + 1
                    
                    # Update progress setiap 10% atau file terakhir
                    if current_processed == 1 or current_processed == num_files or current_processed % max(1, num_files // 10) == 0:
                        split_progress = progress_base + (current_processed / num_files * progress_range)
                        self._notify_progress(
                            progress=int(split_progress), total=100,
                            message=f"Preprocessing {current_processed}/{num_files} gambar",
                            status="info", step=2, split_step=split_step,
                            split=split, current_progress=current_processed, current_total=num_files,
                            total_files_all=total_files_all
                        )
                    
                    # Proses gambar
                    result = preprocess_single_image(
                        img_path, labels_dir, target_images_dir, target_labels_dir,
                        self.pipeline, self.storage, self.file_prefix,
                        preprocessing_options, self.logger
                    )
                    stats['processed' if result else 'skipped'] += 1
                    
                except Exception:
                    stats['failed'] += 1
            
            # Finalisasi statistik
            stats.update({
                'end_time': time.time(), 
                'duration': time.time() - stats['start_time'], 
                'total': num_files, 
                'success': stats['failed'] == 0
            })
            
            self.storage.update_stats(split, stats)
            return stats
                
        except Exception as e:
            return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': str(e)}
    
    def clean_preprocessed(self, split: Optional[str] = None) -> None:
        """Bersihkan hasil preprocessing."""
        if split:
            self.storage.clean_storage(split)
        else:
            [self.storage.clean_storage(split_name) for split_name in DEFAULT_SPLITS]
    
    def get_preprocessed_stats(self) -> Dict[str, Any]:
        """Dapatkan statistik hasil preprocessing."""
        return {split_name: self.storage.get_stats(split_name) 
                for split_name in DEFAULT_SPLITS 
                if self.storage.get_stats(split_name)}
    
    def is_preprocessed(self, split: str) -> bool:
        """Cek apakah split sudah dipreprocess."""
        split_path = self.storage.get_split_path(split)
        return split_path.exists() and len(list(split_path.glob('**/*.jpg'))) > 0