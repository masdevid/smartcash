"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Layanan preprocessing dataset dengan fitur penamaan file berdasarkan kelas dan ID unik
"""

import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import cv2
import numpy as np
from tqdm.auto import tqdm
import uuid
import re

from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.services.preprocessor.pipeline import PreprocessingPipeline
from smartcash.dataset.services.preprocessor.storage import PreprocessedStorage
from smartcash.components.observer import notify, EventTopics

class DatasetPreprocessor:
    """Layanan preprocessing dataset dengan penamaan file berdasarkan kelas dan ID unik."""
    
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
            
            if not (splits_to_process := self._resolve_splits(split)): return {"error": "Tidak ada split valid"}
            start_time, results = time.time(), {}
            
            # Step 1: Persiapan dataset - 10% dari total progress
            self._update_progress(0, len(splits_to_process), f"Mempersiapkan dataset...", step=0, current_step=0, current_total=1)
            
            # Step 2: Proses setiap split - 80% dari total progress
            for i, current_split in enumerate(splits_to_process):
                self._update_progress(i, len(splits_to_process), f"Memproses split {current_split}...", step=1, current_step=1, current_total=3)
                results[current_split] = self._preprocess_split(current_split, force_reprocess, show_progress, **kwargs)
            
            # Step 3: Finalisasi - 10% dari total progress
            self._update_progress(len(splits_to_process), len(splits_to_process), f"Finalisasi hasil preprocessing...", step=2, current_step=2, current_total=3)
            
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
            
            try: notify(event_type=EventTopics.PREPROCESSING_END, sender="dataset_preprocessor", 
                    message=f"Preprocessing selesai: {total_images} gambar berhasil diproses", duration=total_result['processing_time'])
            except Exception: pass
                
            return total_result
                
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan preprocessing: {str(e)}")
            try: notify(event_type=EventTopics.PREPROCESSING_ERROR, sender="dataset_preprocessor", message=f"Error preprocessing: {str(e)}")
            except Exception: pass
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
        """Preprocess satu split dataset dengan tracking progress yang lebih detail."""
        try:
            # Cek apakah sudah dipreprocess dan tidak perlu diproses ulang
            if not force_reprocess and self.is_preprocessed(split):
                self.logger.info(f"✅ Split '{split}' sudah dipreprocess sebelumnya")
                processed = len(list((images_path := self.storage.get_split_path(split) / 'images').glob('*.*'))) if images_path.exists() else 0
                self._update_progress(1, 1, f"Split {split} sudah dipreprocess sebelumnya", step=1, current_step=1, current_total=1)
                return {'processed': processed, 'skipped': 0, 'failed': 0, 'success': True}
                
            # Dapatkan path source dan validasi direktori
            source_dir, images_dir, labels_dir = self._get_source_dir(split), Path(self._get_source_dir(split)) / 'images', Path(self._get_source_dir(split)) / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                error_message = f"❌ Direktori gambar atau label tidak ditemukan: {images_dir} atau {labels_dir}"
                self.logger.error(error_message)
                self._update_progress(0, 1, error_message, status="error", step=1, current_step=1, current_total=1)
                return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': error_message}
                
            # Setup direktori target
            target_dir, target_images_dir, target_labels_dir = self.storage.get_split_path(split), self.storage.get_split_path(split) / 'images', self.storage.get_split_path(split) / 'labels'
            target_images_dir.mkdir(parents=True, exist_ok=True); target_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Mulai preprocessing
            self.logger.info(f"🔄 Memulai preprocessing untuk split '{split}'")
            self._update_progress(0, 100, f"Memulai preprocessing split {split}", step=1, current_step=1, current_total=3)
            
            # Dapatkan daftar file dan validasi
            image_files, num_files = sorted(list(images_dir.glob('*.*'))), len(image_files := sorted(list(images_dir.glob('*.*'))))
            
            if num_files == 0:
                self.logger.warning(f"⚠️ Tidak ada file gambar yang ditemukan di {images_dir}")
                self._update_progress(1, 1, f"Tidak ada file gambar ditemukan di {images_dir}", step=1, current_step=1, current_total=3)
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
            
            # Notifikasi mulai processing gambar
            self._update_progress(0, num_files, f"Memproses {num_files} gambar pada split {split}", step=1, current_step=1, current_total=3)
            
            # Proses setiap gambar
            for i, img_path in enumerate(image_files):
                try:
                    # Update progress callback
                    if self._progress_callback: self._progress_callback(progress=i+1, total=num_files, message=f"Preprocessing {split}: {i+1}/{num_files}", 
                                                                    status='info', current_progress=i+1, current_total=num_files, step=1, 
                                                                    current_step=1, percentage=int((i+1)/num_files*100))
                    
                    stats['processed' if self._preprocess_single_image(img_path, labels_dir, target_images_dir, target_labels_dir, preprocessing_options) else 'skipped'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal memproses {img_path.name}: {str(e)}"); stats['failed'] += 1
                
                progress.update(1)
            
            progress.close()
            
            # Notifikasi finalisasi dan penyimpanan
            self._update_progress(num_files, num_files, f"Menyimpan metadata dan finalisasi split {split}", step=1, current_step=2, current_total=3)
            
            # Update statistik dan simpan metadata
            stats.update({'end_time': time.time(), 'duration': time.time() - stats['start_time'], 'total': num_files, 'success': stats['failed'] == 0})
            self.storage.update_stats(split, stats)
            
            self.logger.info(f"✅ Preprocessing '{split}' selesai: {stats['processed']} berhasil, {stats['skipped']} dilewati, {stats['failed']} gagal, durasi: {stats['duration']:.2f} detik")
            self._update_progress(3, 3, f"Preprocessing split {split} selesai", step=1, current_step=3, current_total=3)
            return stats
                
        except Exception as e:
            self.logger.error(f"❌ Gagal preprocessing split {split}: {str(e)}")
            return {'processed': 0, 'skipped': 0, 'failed': 1, 'success': False, 'error': str(e)}

    def _extract_class_from_label(self, label_path: Path) -> Optional[str]:
        """
        Ekstrak kelas dari file label YOLO.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            Nama kelas atau None jika tidak ditemukan
        """
        if not label_path.exists():
            return None
            
        try:
            # Baca file label
            with open(label_path, 'r') as f:
                label_lines = f.readlines()
                
            # Cek apakah ada isi label
            if not label_lines:
                return None
                
            # Ambil class ID dari line pertama
            first_line = label_lines[0].strip().split()
            if not first_line:
                return None
                
            # Class ID ada di posisi pertama format YOLO
            class_id = first_line[0]
            
            # Map class ID ke nama kelas jika tersedia
            class_names = self.config.get('data', {}).get('class_names', {})
            if class_names and class_id in class_names:
                return class_names[class_id]
                
            # Fallback ke class ID jika nama kelas tidak tersedia
            return f"class{class_id}"
        except Exception as e:
            self.logger.debug(f"⚠️ Gagal ekstrak kelas dari {label_path}: {str(e)}")
            return None
    
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
        Preprocess satu gambar dan label terkait, dan simpan hasilnya dengan penamaan yang ditingkatkan.
        
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
            
            # Ekstrak kelas dari file label untuk penamaan file
            banknote_class = self._extract_class_from_label(label_path)
            
            # Generate ID unik untuk file
            unique_id = str(uuid.uuid4())[:8]  # 8 karakter pertama dari UUID
            
            # Generate nama file baru dengan format {prefix}_{class}_{unique_id}
            new_filename = f"{self.file_prefix}_{banknote_class or 'unknown'}_{unique_id}"
            
            # Simpan hasil gambar
            output_path = target_images_dir / f"{new_filename}.jpg"
            
            # Normalisasi jika perlu sebelum menyimpan
            if normalize and processed_image.dtype == np.float32:
                save_image = (processed_image * 255).astype(np.uint8)
            else:
                save_image = processed_image
                
            cv2.imwrite(str(output_path), save_image)
            
            # Salin file label dengan nama baru
            with open(label_path, 'r') as src_file:
                with open(target_labels_dir / f"{new_filename}.txt", 'w') as dst_file:
                    dst_file.write(src_file.read())
            
            # Simpan metadata
            metadata = {
                'original_path': str(img_path),
                'original_id': img_id,
                'original_size': (image.shape[1], image.shape[0]),  # width, height
                'processed_size': (processed_image.shape[1], processed_image.shape[0]),
                'preprocessing_timestamp': time.time(),
                'banknote_class': banknote_class,
                'new_filename': new_filename
            }
            
            # Simpan ke storage - dengan penanganan error untuk backward compatibility
            try:
                self.storage.save_metadata(split=Path(target_images_dir).parent.name, image_id=new_filename, metadata=metadata)
            except (AttributeError, Exception) as e:
                # Jika metode save_metadata tidak ada (backward compatibility)
                self.logger.debug(f"⚠️ Storage tidak mendukung save_metadata: {str(e)}")
            
            return True
            
        except Exception as e:
            # Tampilkan pesan error dengan nama file yang terpotong
            short_id = img_id
            if len(short_id) > 15:
                short_id = f"...{short_id[-10:]}"
            
            self.logger.error(f"❌ Error saat preprocessing {short_id}: {str(e)}")
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
        
    def _update_progress(self, current: int, total: int, message: str = None, status: str = 'info', **kwargs) -> None:
        """Update progress dengan callback dan parameter tambahan untuk step-tracking."""
        if self._progress_callback: self._progress_callback(progress=current, total=total, message=message, status=status, **kwargs)
        
        try: notify(event_type=EventTopics.PREPROCESSING_PROGRESS, sender="dataset_preprocessor", 
                message=message or f"Preprocessing progress: {int(current/total*100)}%" if total > 0 else "Preprocessing progress", 
                progress=current, total=total, **kwargs)
        except Exception: pass

# Class alias untuk backward compatibility
DatasetPreprocessorService = DatasetPreprocessor