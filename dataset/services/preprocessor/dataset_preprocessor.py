"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Layanan preprocessing dataset dan penyimpanan hasil untuk penggunaan berikutnya
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

class DatasetPreprocessor:
    """Layanan preprocessing dataset dan penyimpanan hasil untuk penggunaan berikutnya."""
    
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
        
    def register_progress_callback(self, callback: Callable) -> None:
        """
        Register progress callback untuk tracking progress.
        
        Args:
            callback: Progress callback function
        """
        self._progress_callback = callback
        
    def preprocess_dataset(self, split: str = None, force_reprocess: bool = False, 
                         show_progress: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Preprocess dataset dan simpan hasil untuk penggunaan berikutnya.
        
        Args:
            split: Split dataset ('train', 'valid', 'test', None untuk semua split)
            force_reprocess: Paksa untuk memproses ulang, meskipun sudah ada hasil sebelumnya
            show_progress: Tampilkan progress bar
            **kwargs: Parameter tambahan untuk preprocessing
            
        Returns:
            Dictionary dengan hasil preprocessing
        """
        try:
            # Notifikasi observer
            try:
                notify(
                    event_type=EventTopics.PREPROCESSING_START,
                    sender="dataset_preprocessor",
                    message=f"Memulai preprocessing dataset {split or 'semua split'}"
                )
            except Exception:
                pass
            
            splits_to_process = self._resolve_splits(split)
            
            if not splits_to_process:
                self.logger.warning(f"⚠️ Tidak ada split valid untuk diproses")
                return {"error": "Tidak ada split valid"}
                
            start_time = time.time()
            results = {}
            
            # Total progress tracking
            total_splits = len(splits_to_process)
            for i, current_split in enumerate(splits_to_process):
                # Update progress
                self._update_progress(i, total_splits, f"Memproses split {current_split}...")
                
                # Proses split saat ini
                split_result = self._preprocess_split(current_split, force_reprocess, show_progress, **kwargs)
                results[current_split] = split_result
                
                # Update progress
                self._update_progress(i+1, total_splits, f"Selesai split {current_split}")
            
            # Hitung total statistik
            total_images = sum(r.get('processed', 0) for r in results.values())
            total_skipped = sum(r.get('skipped', 0) for r in results.values())
            total_failed = sum(r.get('failed', 0) for r in results.values())
            
            # Generate hasil
            total_result = {
                'total_images': total_images,
                'total_skipped': total_skipped,
                'total_failed': total_failed,
                'split_stats': {s: {
                    'images': r.get('processed', 0),
                    'labels': r.get('processed', 0),
                    'complete': r.get('processed', 0) > 0 and r.get('failed', 0) == 0
                } for s, r in results.items()},
                'processing_time': time.time() - start_time,
                'success': total_failed == 0,
                'output_dir': str(self.storage.output_dir),
                'image_size': self.config.get('preprocessing', {}).get('img_size', [640, 640])
            }
            
            # Notifikasi observer
            try:
                notify(
                    event_type=EventTopics.PREPROCESSING_END,
                    sender="dataset_preprocessor",
                    message=f"Preprocessing selesai: {total_images} gambar berhasil diproses",
                    duration=total_result['processing_time']
                )
            except Exception:
                pass
                
            return total_result
                
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan preprocessing: {str(e)}")
            
            # Notifikasi observer
            try:
                notify(
                    event_type=EventTopics.PREPROCESSING_ERROR,
                    sender="dataset_preprocessor",
                    message=f"Error preprocessing: {str(e)}"
                )
            except Exception:
                pass
                
            raise DatasetProcessingError(f"Gagal melakukan preprocessing: {str(e)}")
    
    def _resolve_splits(self, split: Optional[str]) -> List[str]:
        """
        Resolve split parameter menjadi list split yang akan diproses.
        
        Args:
            split: Split yang akan diproses ('train', 'valid', 'test', None untuk semua)
            
        Returns:
            List split yang akan diproses
        """
        # Jika split None atau 'all', proses semua split
        if not split or split.lower() == 'all':
            return ['train', 'valid', 'test']
        # Untuk split 'val', gunakan 'valid'
        elif split.lower() == 'val':
            return ['valid']
        # Untuk nilai split lainnya
        else:
            return [split]
    
    def _preprocess_split(self, split: str, force_reprocess: bool, show_progress: bool, **kwargs) -> Dict[str, Any]:
        """
        Preprocess satu split dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            force_reprocess: Paksa untuk memproses ulang
            show_progress: Tampilkan progress bar
            **kwargs: Parameter tambahan untuk preprocessing
            
        Returns:
            Dictionary berisi hasil preprocessing split
        """
        try:
            # Cek apakah sudah dipreprocess dan tidak perlu diproses ulang
            if not force_reprocess and self.is_preprocessed(split):
                self.logger.info(f"✅ Split '{split}' sudah dipreprocess sebelumnya")
                # Hitung jumlah gambar
                split_path = self.storage.get_split_path(split)
                images_path = split_path / 'images'
                processed = len(list(images_path.glob('*.*'))) if images_path.exists() else 0
                return {'processed': processed, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Dapatkan path source data
            source_dir = self._get_source_dir(split)
            images_dir = Path(source_dir) / 'images'
            labels_dir = Path(source_dir) / 'labels'
            
            # Validasi direktori source data
            if not images_dir.exists() or not labels_dir.exists():
                error_message = f"❌ Direktori gambar atau label tidak ditemukan: {images_dir} atau {labels_dir}"
                self.logger.error(error_message)
                return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': error_message}
                
            # Dapatkan path target
            target_dir = self.storage.get_split_path(split)
            target_images_dir = target_dir / 'images'
            target_labels_dir = target_dir / 'labels'
            
            # Buat direktori output jika belum ada
            target_images_dir.mkdir(parents=True, exist_ok=True)
            target_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Mulai preprocessing
            self.logger.info(f"🔄 Memulai preprocessing untuk split '{split}'")
            
            # Dapatkan daftar file gambar
            image_files = sorted(list(images_dir.glob('*.*')))
            num_files = len(image_files)
            
            if num_files == 0:
                self.logger.warning(f"⚠️ Tidak ada file gambar yang ditemukan di {images_dir}")
                return {'processed': 0, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Setup progress tracking
            progress = tqdm(total=num_files, desc=f"Preprocessing {split}", disable=not show_progress)
            
            stats = {
                'processed': 0,
                'skipped': 0,
                'failed': 0,
                'start_time': time.time()
            }
            
            # Extract image size dan normalisasi dari config
            img_size = self.config.get('preprocessing', {}).get('img_size', [640, 640])
            normalize = self.config.get('preprocessing', {}).get('normalization', {}).get('enabled', True)
            preserve_aspect_ratio = self.config.get('preprocessing', {}).get('normalization', {}).get('preserve_aspect_ratio', True)
            
            # Extract preprocessing options dari kwargs jika ada
            preprocessing_options = {
                'img_size': kwargs.get('img_size', img_size),
                'normalize': kwargs.get('normalize', normalize),
                'preserve_aspect_ratio': kwargs.get('preserve_aspect_ratio', preserve_aspect_ratio)
            }
            
            # Proses setiap gambar
            for i, img_path in enumerate(image_files):
                try:
                    # Update progress callback untuk current progress
                    if self._progress_callback:
                        self._progress_callback(
                            progress=stats['processed'],
                            total=num_files,
                            message=f"Preprocessing {split}: {i+1}/{num_files}",
                            status='info',
                            current_progress=i,
                            current_total=num_files
                        )
                    
                    result = self._preprocess_single_image(
                        img_path, labels_dir, target_images_dir, target_labels_dir, 
                        preprocessing_options=preprocessing_options
                    )
                    
                    if result:
                        stats['processed'] += 1
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal memproses {img_path.name}: {str(e)}")
                    stats['failed'] += 1
                
                progress.update(1)
            
            progress.close()
            
            # Update statistik
            stats['end_time'] = time.time()
            stats['duration'] = stats['end_time'] - stats['start_time']
            stats['total'] = num_files
            stats['success'] = stats['failed'] == 0
            
            # Simpan metadata
            self.storage.update_stats(split, stats)
            
            self.logger.info(
                f"✅ Preprocessing '{split}' selesai: {stats['processed']} berhasil, "
                f"{stats['skipped']} dilewati, {stats['failed']} gagal, "
                f"durasi: {stats['duration']:.2f} detik"
            )
                
            return stats
                
        except Exception as e:
            self.logger.error(f"❌ Gagal preprocessing split {split}: {str(e)}")
            return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': str(e)}
    
    def _get_source_dir(self, split: str) -> str:
        """
        Dapatkan direktori sumber data split.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Path direktori sumber data
        """
        # Cek apakah ada path spesifik untuk split
        data_dir = self.config.get('data', {}).get('dir', 'data')
        
        # Jika punya local.split, gunakan itu
        if 'local' in self.config.get('data', {}) and split in self.config.get('data', {}).get('local', {}):
            return self.config['data']['local'][split]
        
        # Fallback ke direktori default
        return os.path.join(data_dir, split)
        
    def _preprocess_single_image(self, img_path: Path, labels_dir: Path, 
                              target_images_dir: Path, target_labels_dir: Path,
                              preprocessing_options: Dict[str, Any] = None) -> bool:
        """
        Preprocess satu gambar dan label terkait, dan simpan hasilnya.
        
        Args:
            img_path: Path ke file gambar
            labels_dir: Direktori berisi file label
            target_images_dir: Direktori output untuk gambar
            target_labels_dir: Direktori output untuk label
            preprocessing_options: Opsi preprocessing tambahan
            
        Returns:
            Boolean menunjukkan keberhasilan preprocessing
        """
        # Generate nama file output berdasarkan nama file input
        img_id = img_path.stem
        label_path = labels_dir / f"{img_id}.txt"
        
        # Lewati jika tidak ada file label yang sesuai
        if not label_path.exists():
            self.logger.debug(f"⚠️ File label tidak ditemukan untuk {img_id}, dilewati")
            return False
            
        # Proses gambar
        try:
            # Baca gambar
            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.warning(f"⚠️ Tidak dapat membaca gambar: {img_path}")
                return False
                
            # Ambil opsi preprocessing dari options
            options = preprocessing_options or {}
            img_size = options.get('img_size', [640, 640])
            normalize = options.get('normalize', True)
            preserve_aspect_ratio = options.get('preserve_aspect_ratio', True)
            
            # Setup preprocessing options
            self.pipeline.set_options(img_size=img_size, normalize=normalize, 
                                     preserve_aspect_ratio=preserve_aspect_ratio)
            
            # Preprocess gambar
            processed_image = self.pipeline.process(image)
            
            # Simpan hasil gambar
            output_path = target_images_dir / f"{img_id}.jpg"
            
            # Normalisasi jika perlu sebelum menyimpan
            if normalize and processed_image.dtype == np.float32:
                save_image = (processed_image * 255).astype(np.uint8)
            else:
                save_image = processed_image
                
            cv2.imwrite(str(output_path), save_image)
            
            # Salin file label tanpa preprocessing (format YOLO)
            import shutil
            shutil.copy2(label_path, target_labels_dir / f"{img_id}.txt")
            
            # Simpan metadata
            metadata = {
                'original_path': str(img_path),
                'original_size': (image.shape[1], image.shape[0]),  # width, height
                'processed_size': (processed_image.shape[1], processed_image.shape[0]),
                'preprocessing_timestamp': time.time()
            }
            
            # Simpan ke storage
            self.storage.save_metadata(split=Path(target_images_dir).parent.name, image_id=img_id, metadata=metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat preprocessing {img_id}: {str(e)}")
            return False
    
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
        
    def _update_progress(self, current: int, total: int, message: str = None) -> None:
        """
        Update progress dengan callback.
        
        Args:
            current: Nilai progress saat ini
            total: Total nilai progress
            message: Pesan progress (opsional)
        """
        if self._progress_callback:
            self._progress_callback(
                progress=current,
                total=total,
                message=message,
                status='info'
            )
            
        # Notifikasi observer
        try:
            notify(
                event_type=EventTopics.PREPROCESSING_PROGRESS,
                sender="dataset_preprocessor",
                message=message or f"Preprocessing progress: {int(current/total*100)}%",
                progress=current,
                total=total
            )
        except Exception:
            pass

# Class alias untuk backward compatibility
DatasetPreprocessorService = DatasetPreprocessor