"""
File: smartcash/dataset/services/preprocessor/dataset_preprocessor.py
Deskripsi: Layanan untuk preprocessing dataset dan menyimpan hasilnya untuk penggunaan berikutnya
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm import tqdm

from smartcash.common.logger import get_logger
from smartcash.model.utils.preprocessing_model_utils import ModelPreprocessor, letterbox
from smartcash.dataset.utils.file.file_processor import FileProcessor
from smartcash.common.config import get_config_manager

class DatasetPreprocessor:
    """
    Layanan untuk melakukan preprocessing pada dataset dan menyimpan hasilnya
    untuk mengurangi overhead komputasi saat training dan evaluasi.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi DatasetPreprocessor.
        
        Args:
            config: Konfigurasi untuk preprocessor
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger("dataset_preprocessor")
        self.config_manager = get_config_manager()
        
        # Default config
        self.default_config = {
            'img_size': (640, 640),
            'preprocessed_dir': 'data/preprocessed',
            'dataset_dir': 'data/dataset',
            'max_workers': 8,
            'preserve_labels': True,
            'use_letterbox': True,
            'normalize': True
        }
        
        # Merge konfigurasi
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Inisialisasi model preprocessor
        self.model_preprocessor = ModelPreprocessor(
            img_size=self.config['img_size'],
            pad_to_square=self.config['use_letterbox']
        )
        
        # Setup paths
        self.dataset_dir = Path(self.config['dataset_dir'])
        self.preprocessed_dir = Path(self.config['preprocessed_dir'])
        
        self.logger.info(f"🛠️ DatasetPreprocessor diinisialisasi (target dir: {self.preprocessed_dir})")
    
    def preprocess_dataset(
        self,
        split: str = 'all',
        force_reprocess: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Proses dataset dan simpan hasil preprocessing.
        
        Args:
            split: Split dataset ('train', 'val', 'test', 'all')
            force_reprocess: Paksa proses ulang meskipun sudah ada
            show_progress: Tampilkan progress bar
            
        Returns:
            Statistik hasil preprocessing
        """
        splits_to_process = ['train', 'val', 'test'] if split == 'all' else [split]
        results = {}
        
        for current_split in splits_to_process:
            # Validasi split
            source_dir = self.dataset_dir / current_split
            if not source_dir.exists():
                self.logger.warning(f"⚠️ Split {current_split} tidak ditemukan di {source_dir}")
                continue
            
            # Target directory
            target_dir = self.preprocessed_dir / current_split
            
            # Cek apakah perlu membersihkan terlebih dahulu
            if force_reprocess and target_dir.exists():
                self.logger.info(f"🧹 Membersihkan direktori preprocessed yang sudah ada: {target_dir}")
                shutil.rmtree(target_dir)
            
            # Buat direktori target jika belum ada
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Buat subdirektori untuk gambar dan label
            images_dir = source_dir / 'images'
            labels_dir = source_dir / 'labels'
            
            # Buat direktori target untuk gambar dan label
            target_images_dir = target_dir / 'images'
            target_labels_dir = target_dir / 'labels'
            
            target_images_dir.mkdir(exist_ok=True)
            if self.config['preserve_labels']:
                target_labels_dir.mkdir(exist_ok=True)
            
            # Dapatkan daftar gambar untuk diproses
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
            
            if not image_files:
                self.logger.warning(f"⚠️ Tidak ada gambar di {images_dir}")
                continue
                
            # Setup progress bar
            total_files = len(image_files)
            self.logger.info(f"🔍 Menemukan {total_files} gambar untuk diproses di {current_split}")
            
            if show_progress:
                pbar = tqdm(total=total_files, desc=f"🔄 Preprocessing {current_split}")
            
            # Buat worker pool
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # Submit semua tugas
                futures = []
                for img_path in image_files:
                    future = executor.submit(
                        self._preprocess_single_image,
                        img_path,
                        labels_dir,
                        target_images_dir,
                        target_labels_dir
                    )
                    futures.append(future)
                
                # Proses hasil
                successful = 0
                failed = 0
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        self.logger.error(f"❌ Error saat preprocessing: {str(e)}")
                        failed += 1
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
                
            self.logger.success(
                f"✅ Preprocessing {current_split} selesai: "
                f"{successful} berhasil, {failed} gagal"
            )
            
            results[current_split] = {
                'total': total_files,
                'successful': successful,
                'failed': failed
            }
        
        return results
    
    def _preprocess_single_image(
        self,
        img_path: Path,
        labels_dir: Path,
        target_images_dir: Path,
        target_labels_dir: Path
    ) -> bool:
        """
        Preprocessing satu gambar dan label terkait.
        
        Args:
            img_path: Path gambar
            labels_dir: Direktori label
            target_images_dir: Direktori target untuk gambar
            target_labels_dir: Direktori target untuk label
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Target path
            target_img_path = target_images_dir / img_path.name
            
            # Skip jika sudah ada dan tidak dipaksa reprocess
            if target_img_path.exists():
                return True
            
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Konversi BGR ke RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ukuran asli
            original_height, original_width = img.shape[:2]
            
            # Lakukan letterbox resize
            if self.config['use_letterbox']:
                img_resized, ratio, pad = letterbox(
                    img_rgb, 
                    self.config['img_size'], 
                    auto=False, 
                    stride=32
                )
            else:
                # Atau resize langsung
                img_resized = cv2.resize(img_rgb, self.config['img_size'])
                ratio = (self.config['img_size'][0] / original_width, 
                        self.config['img_size'][1] / original_height)
                pad = (0, 0)
                
            # Simpan metadata transformasi untuk referensi
            metadata = {
                'original_size': (original_width, original_height),
                'new_size': self.config['img_size'],
                'ratio': ratio,
                'padding': pad
            }
            
            if self.config['normalize']:
                # Normalisasi (simpan sebagai float16 untuk efisiensi penyimpanan)
                img_preprocessed = img_resized.astype(np.float32) / 255.0
                
                # Simpan preprocessing hasil (numpy array)
                np.save(str(target_img_path).replace(target_img_path.suffix, '.npy'), 
                       img_preprocessed)
                
                # Simpan metadata
                np.save(str(target_img_path).replace(target_img_path.suffix, '_meta.npy'), 
                       np.array([metadata], dtype=object))
            else:
                # Simpan sebagai gambar saja jika tidak perlu normalisasi
                cv2.imwrite(str(target_img_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            
            # Salin label jika perlu
            if self.config['preserve_labels']:
                label_path = labels_dir / img_path.with_suffix('.txt').name
                if label_path.exists():
                    target_label_path = target_labels_dir / label_path.name
                    shutil.copy2(label_path, target_label_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat preprocessing {img_path.name}: {str(e)}")
            return False
    
    def clean_preprocessed(self, split: str = 'all') -> bool:
        """
        Bersihkan hasil preprocessing.
        
        Args:
            split: Split dataset yang akan dibersihkan ('train', 'val', 'test', 'all')
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            if split == 'all':
                if self.preprocessed_dir.exists():
                    self.logger.info(f"🧹 Membersihkan semua data preprocessed: {self.preprocessed_dir}")
                    shutil.rmtree(self.preprocessed_dir)
                    self.preprocessed_dir.mkdir(parents=True)
                return True
            else:
                target_dir = self.preprocessed_dir / split
                if target_dir.exists():
                    self.logger.info(f"🧹 Membersihkan data preprocessed untuk {split}: {target_dir}")
                    shutil.rmtree(target_dir)
                    target_dir.mkdir(parents=True)
                return True
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan data preprocessed: {str(e)}")
            return False
    
    def get_preprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik hasil preprocessing.
        
        Returns:
            Statistik data preprocessed
        """
        stats = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.preprocessed_dir / split
            if not split_dir.exists():
                stats[split] = {'images': 0, 'labels': 0}
                continue
            
            # Hitung gambar
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            n_images = len(list(images_dir.glob('*.npy'))) + len(list(images_dir.glob('*.jpg'))) + \
                       len(list(images_dir.glob('*.jpeg'))) + len(list(images_dir.glob('*.png')))
            
            n_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            stats[split] = {
                'images': n_images,
                'labels': n_labels
            }
        
        return stats
    
    def is_preprocessed(self, split: str) -> bool:
        """
        Cek apakah split dataset sudah dipreprocessing.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            
        Returns:
            True jika sudah dipreprocessing, False jika belum
        """
        split_dir = self.preprocessed_dir / split
        if not split_dir.exists():
            return False
        
        images_dir = split_dir / 'images'
        if not images_dir.exists():
            return False
        
        # Cek apakah ada minimal 1 file
        return len(list(images_dir.glob('*.*'))) > 0