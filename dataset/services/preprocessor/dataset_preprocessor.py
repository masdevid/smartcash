"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Layanan preprocessing dataset dan penyimpanan hasil untuk penggunaan berikutnya
"""

import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.services.preprocessor.pipeline import PreprocessingPipeline
from smartcash.dataset.services.preprocessor.storage import PreprocessedStorage

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
        
        # Inisialisasi pipeline preprocessing
        self.pipeline = PreprocessingPipeline(config, self.logger)
        
        # Setup direktori output
        output_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        self.storage = PreprocessedStorage(output_dir, self.logger)
        
    def preprocess_dataset(self, split: str, force_reprocess: bool = False, 
                         show_progress: bool = True) -> bool:
        """
        Preprocess dataset dan simpan hasil untuk penggunaan berikutnya.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            force_reprocess: Paksa untuk memproses ulang, meskipun sudah ada hasil sebelumnya
            show_progress: Tampilkan progress bar
            
        Returns:
            Boolean menunjukkan keberhasilan preprocessing
        """
        try:
            # Cek apakah sudah dipreprocess dan tidak perlu diproses ulang
            if not force_reprocess and self.is_preprocessed(split):
                self.logger.info(f"✅ Split '{split}' sudah dipreprocess sebelumnya")
                return True
                
            # Dapatkan path source data
            data_dirs = self.config.get('data', {}).get('local', {})
            if not data_dirs or split not in data_dirs:
                raise DatasetProcessingError(f"❌ Direktori untuk split '{split}' tidak ditemukan dalam konfigurasi")
                
            source_dir = data_dirs[split]
            images_dir = Path(source_dir) / 'images'
            labels_dir = Path(source_dir) / 'labels'
            
            # Validasi direktori source data
            if not images_dir.exists() or not labels_dir.exists():
                raise DatasetProcessingError(
                    f"❌ Direktori gambar atau label tidak ditemukan: {images_dir} atau {labels_dir}")
                
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
                return False
                
            # Setup progress tracking
            from tqdm.auto import tqdm
            progress = tqdm(total=num_files, desc=f"Preprocessing {split}", disable=not show_progress)
            
            stats = {
                'processed': 0,
                'skipped': 0,
                'failed': 0,
                'start_time': time.time()
            }
            
            # Proses setiap gambar
            for img_path in image_files:
                try:
                    result = self._preprocess_single_image(
                        img_path, labels_dir, target_images_dir, target_labels_dir)
                    
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
            
            # Simpan metadata
            self.storage.update_stats(split, stats)
            
            self.logger.info(
                f"✅ Preprocessing '{split}' selesai: {stats['processed']} berhasil, "
                f"{stats['skipped']} dilewati, {stats['failed']} gagal, "
                f"durasi: {stats['duration']:.2f} detik")
                
            return stats['failed'] == 0
                
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan preprocessing: {str(e)}")
            raise DatasetProcessingError(f"Gagal melakukan preprocessing: {str(e)}")
    
    def _preprocess_single_image(self, img_path: Path, labels_dir: Path, 
                              target_images_dir: Path, target_labels_dir: Path) -> bool:
        """
        Preprocess satu gambar dan label terkait, dan simpan hasilnya.
        
        Args:
            img_path: Path ke file gambar
            labels_dir: Direktori berisi file label
            target_images_dir: Direktori output untuk gambar
            target_labels_dir: Direktori output untuk label
            
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
        import cv2
        import numpy as np
        
        try:
            # Baca gambar
            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.warning(f"⚠️ Tidak dapat membaca gambar: {img_path}")
                return False
                
            # Preprocess gambar
            processed_image = self.pipeline.process(image)
            
            # Simpan hasil gambar
            output_path = target_images_dir / f"{img_id}.jpg"
            cv2.imwrite(str(output_path), processed_image)
            
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
            self.storage.save_preprocessed_image(
                split='temp', image_id=img_id, 
                image_data=processed_image, metadata=metadata)
            
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

# Class alias untuk backward compatibility
DatasetPreprocessorService = DatasetPreprocessor